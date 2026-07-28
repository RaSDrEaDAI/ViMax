import logging
import requests
from tenacity import retry


def concatenate_shot_videos(
    video_paths,
    out_path,
    codec="libx264",
    preset="medium",
):
    """Concatenate shot videos into one file and write it to ``out_path``.

    The ONE place the final concat is written, so the ``logger=None`` below
    cannot regress in one pipeline while staying fixed in the other.

    WHY logger=None (this is load-bearing, not cosmetic): moviepy defaults to
    ``logger="bar"``, which builds a proglog/tqdm progress bar that writes to
    ``sys.stdout`` and flushes on every update. Under the orchestrator bridge,
    stdout is a STRUCTURED channel — interleaved progress plus a final
    ``VIMAX_RESULT `` line, consumed through a pipe — and on Windows that flush
    raises ``OSError: [Errno 22] Invalid argument``. It killed a 62-minute run at
    the very last step with every shot's video.mp4 already on disk, reporting
    ``final_video: null`` (.working_dir/orchestrator/b368d288). The traceback
    runs write_videofile -> write_audiofile -> ffmpeg_audiowrite -> iter_chunks
    -> proglog -> tqdm.status_printer -> sys.stdout.flush(). A progress bar is
    worthless in a piped run regardless.

    ``codec``/``preset`` match moviepy's own effective defaults for .mp4
    (codec=None infers libx264; preset is already "medium"), so naming them is
    explicit rather than a change in output.
    """
    # Imported here, not at module scope: utils.video is imported by code paths
    # that never concatenate, and moviepy is a heavy import.
    from moviepy import VideoFileClip, concatenate_videoclips

    clips = [VideoFileClip(path) for path in video_paths]
    try:
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(
            out_path, codec=codec, preset=preset, logger=None,
        )
    finally:
        # Release the ffmpeg readers even if the write fails. Without this a
        # failed concat leaves file handles open, and on Windows that blocks the
        # retry from overwriting the very files it needs.
        for clip in clips:
            try:
                clip.close()
            except Exception:  # noqa: BLE001
                pass
    return out_path


@retry
def download_video(url, save_path):
    try:
        logging.info(f"Downloading video from {url} to {save_path}")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
    
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Video downloaded successfully to {save_path}")
    
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        raise e
