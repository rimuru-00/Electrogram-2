from __future__ import annotations

import asyncio
import functools
import inspect
import io
import logging
import math
import time
from hashlib import md5
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, BinaryIO, overload

import pyrogram
from pyrogram import StopTransmissionError, raw
from pyrogram.session import Session

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

# How long (seconds) before a cached upload session is rebuilt unconditionally,
# regardless of health check result. Matches get_file TTL.
_UPLOAD_SESSION_TTL = 21600

# Timeout (seconds) for the health-check Ping.
# A live connection responds in milliseconds; anything longer means the
# TCP socket is silently dead (common on Heroku between jobs).
_HEALTH_CHECK_TIMEOUT = 5.0


class SaveFile:
    @staticmethod
    async def _session_is_healthy(session: Session) -> bool:
        """
        Two-stage health check before reusing a cached upload session.

        Stage 1 — is_started flag (free, no network):
            If False the session is stopped or mid-restart — already dead.

        Stage 2 — real Ping with short timeout (one round trip):
            Catches silently dropped TCP connections where the session object
            still thinks it's alive but the socket is gone. Common on Heroku
            when the bot is quiet between uploads.

        Returns True only when both stages pass.
        """
        # Stage 1
        if not session.is_started.is_set():
            log.debug("Upload session health: is_started=False — dead")
            return False

        # Stage 2
        try:
            await asyncio.wait_for(
                session.invoke(raw.functions.Ping(ping_id=0)),
                timeout=_HEALTH_CHECK_TIMEOUT,
            )
            log.debug("Upload session health: ping OK — healthy")
            return True
        except Exception as e:
            log.debug("Upload session health: ping failed (%s) — dead", e)
            return False

    @overload
    async def save_file(
        self: pyrogram.Client,
        path: str | BinaryIO,
        file_id: int | None = None,
        file_part: int = 0,
        progress: Callable | None = None,
        progress_args: tuple = (),
    ) -> raw.base.InputFile: ...

    @overload
    async def save_file(
        self: pyrogram.Client,
        path: None,
        file_id: int | None = None,
        file_part: int = 0,
        progress: Callable | None = None,
        progress_args: tuple = (),
    ) -> None: ...

    async def save_file(
        self: pyrogram.Client,
        path: str | BinaryIO | None,
        file_id: int | None = None,
        file_part: int = 0,
        progress: Callable | None = None,
        progress_args: tuple = (),
    ) -> raw.base.InputFile | None:
        """Upload a file onto Telegram servers, without actually sending the message to anyone.
        Useful whenever an InputFile type is required.

        .. note::
            This is a utility method intended to be used **only** when working with raw
            :obj:`functions <pyrogram.api.functions>` (i.e: a Telegram API method you wish to use which is not
            available yet in the Client class as an easy-to-use method).
        .. include:: /_includes/usable-by/users-bots.rst

        Parameters:
            path (``str`` | ``BinaryIO``):
                The path of the file you want to upload that exists on your local machine or a binary
                file-like object with its attribute ".name" set for in-memory uploads.

            file_id (``int``, *optional*):
                In case a file part expired, pass the file_id and the file_part to retry uploading
                that specific chunk.

            file_part (``int``, *optional*):
                In case a file part expired, pass the file_id and the file_part to retry uploading
                that specific chunk.

            progress (``Callable``, *optional*):
                Pass a callback function to view the file transmission progress.
                The function must take *(current, total)* as positional arguments and will be called
                back each time a new file chunk has been successfully transmitted.

            progress_args (``tuple``, *optional*):
                Extra custom arguments for the progress callback function.

        Returns:
            ``InputFile``: On success, the uploaded file is returned in form of an InputFile object.

        Raises:
            RPCError: In case of a Telegram RPC error.
        """
        if path is None:
            return None

        async with self.save_file_semaphore:
            # ── Worker: sends one chunk, retries 3× with exponential backoff ──
            async def worker(session: Session) -> None:
                while True:
                    data = await queue.get()
                    if data is None:
                        return
                    for attempt in range(3):
                        try:
                            await session.invoke(data)
                            break
                        except Exception as e:
                            log.warning(
                                "[%s] Upload chunk retry %s/3 — %s",
                                self.name,
                                attempt + 1,
                                e,
                            )
                            await asyncio.sleep(2 ** attempt)

            def create_rpc(chunk, file_part, is_big, file_id, file_total_parts):
                if is_big:
                    return raw.functions.upload.SaveBigFilePart(
                        file_id=file_id,
                        file_part=file_part,
                        file_total_parts=file_total_parts,
                        bytes=chunk,
                    )
                return raw.functions.upload.SaveFilePart(
                    file_id=file_id,
                    file_part=file_part,
                    bytes=chunk,
                )

            part_size = 512 * 1024
            queue: asyncio.Queue = asyncio.Queue(32)

            with (
                Path(path).open("rb", buffering=part_size)  # noqa: ASYNC230
                if isinstance(path, str | PurePath)
                else path
            ) as fp:
                file_name = getattr(fp, "name", "file.jpg")
                fp.seek(0, io.SEEK_END)
                file_size = fp.tell()
                fp.seek(0)

                if file_size == 0:
                    raise ValueError("File size equals to 0 B")

                file_size_limit_mib = 4000 if self.me.is_premium else 2000
                if file_size > file_size_limit_mib * 1024 * 1024:
                    raise ValueError(
                        f"Can't upload files bigger than {file_size_limit_mib} MiB",
                    )

                file_total_parts = math.ceil(file_size / part_size)
                is_big = file_size > 10 * 1024 * 1024
                workers_count = 4 if is_big else 1
                is_missing_part = file_id is not None

                if file_id is None:
                    file_id = self.rnd_id()

                md5_obj = md5() if not is_big and not is_missing_part else None

                # ── Acquire a healthy cached upload session ───────────────────
                # Uploads always go to the home DC, so dc_id is always the same.
                # Kept separate from media_sessions so a long upload never
                # evicts a download session and vice versa.
                dc_id = await self.storage.dc_id()

                async with self.upload_sessions_lock:
                    session = self.upload_sessions.get(dc_id)
                    now = time.time()
                    session_age = now - self.upload_sessions_timestamps.get(dc_id, 0)

                    if session is not None:
                        if session_age > _UPLOAD_SESSION_TTL:
                            # TTL expired — rebuild unconditionally
                            log.info(
                                "[%s] Upload session DC%s age=%.0fs >= TTL=%ss — rebuilding",
                                self.name, dc_id, session_age, _UPLOAD_SESSION_TTL,
                            )
                            try:
                                await session.stop()
                            except Exception:
                                pass
                            session = None
                        else:
                            # Within TTL — health check before trusting it
                            healthy = await self._session_is_healthy(session)
                            if healthy:
                                log.debug(
                                    "[%s] Reusing upload session DC%s (age=%.0fs)",
                                    self.name, dc_id, session_age,
                                )
                            else:
                                log.warning(
                                    "[%s] Upload session DC%s failed health check — rebuilding",
                                    self.name, dc_id,
                                )
                                try:
                                    await session.stop()
                                except Exception:
                                    pass
                                session = None

                    if session is None:
                        session = Session(
                            self,
                            dc_id,
                            await self.storage.auth_key(),
                            await self.storage.test_mode(),
                            is_media=True,
                        )
                        await session.start()
                        self.upload_sessions[dc_id] = session
                        self.upload_sessions_timestamps[dc_id] = time.time()
                        log.info(
                            "[%s] Upload session DC%s created and cached",
                            self.name, dc_id,
                        )
                # ── End session acquisition ───────────────────────────────────

                # All workers share the single cached session.
                # Multiple concurrent invoke() calls on one MTProto session
                # is safe — multiplexed internally via msg_id.
                task_workers = [
                    self.loop.create_task(worker(session))
                    for _ in range(workers_count)
                ]

                try:
                    fp.seek(part_size * file_part)
                    next_chunk_task = self.loop.create_task(self.preload(fp, part_size))

                    md5_checksum = ""

                    while True:
                        chunk = await next_chunk_task
                        next_chunk_task = self.loop.create_task(
                            self.preload(fp, part_size),
                        )

                        if not chunk:
                            if md5_obj:
                                md5_checksum = md5_obj.hexdigest()
                            break

                        await queue.put(
                            create_rpc(chunk, file_part, is_big, file_id, file_total_parts),
                        )

                        if is_missing_part:
                            return None

                        if md5_obj:
                            md5_obj.update(chunk)

                        file_part += 1

                        if progress:
                            func = functools.partial(
                                progress,
                                min(file_part * part_size, file_size),
                                file_size,
                                *progress_args,
                            )
                            if inspect.iscoroutinefunction(progress):
                                await func()
                            else:
                                await self.loop.run_in_executor(self.executor, func)

                except StopTransmissionError:
                    raise
                except Exception as e:
                    log.error("[%s] Upload failed at part %s: %s", self.name, file_part, e)
                    # Session broke mid-transfer — evict so next upload gets a fresh one
                    async with self.upload_sessions_lock:
                        if self.upload_sessions.get(dc_id) is session:
                            log.warning(
                                "[%s] Evicting broken upload session DC%s from cache",
                                self.name, dc_id,
                            )
                            self.upload_sessions.pop(dc_id, None)
                            self.upload_sessions_timestamps.pop(dc_id, None)
                            try:
                                await session.stop()
                            except Exception:
                                pass
                else:
                    if is_big:
                        return raw.types.InputFileBig(
                            id=file_id,
                            parts=file_total_parts,
                            name=file_name,
                        )
                    return raw.types.InputFile(
                        id=file_id,
                        parts=file_total_parts,
                        name=file_name,
                        md5_checksum=md5_checksum,
                    )
                finally:
                    # Signal workers to exit and wait cleanly.
                    # Session is NOT stopped — stays cached for next upload.
                    for _ in task_workers:
                        await queue.put(None)
                    await asyncio.gather(*task_workers)

    async def preload(self, fp, part_size: int) -> bytes:
        return fp.read(part_size)
