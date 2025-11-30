# -*- coding: utf-8 -*-
"""Pipeline runner using subprocess.

Provides functionality to run the Kedro pipeline in a subprocess
with streaming output for real-time feedback in the GUI.
"""
import subprocess
import threading
from pathlib import Path
from typing import Generator, Optional
import os


class PipelineRunner:
    """Runs kedro pipeline in subprocess with streaming output."""

    def __init__(self, project_path: Path = None):
        """Initialize PipelineRunner.

        Args:
            project_path: Path to the Kedro project root. If None, uses parent of gui/.
        """
        if project_path is None:
            project_path = Path(__file__).parent.parent
        self.project_path = Path(project_path)
        self.process: Optional[subprocess.Popen] = None
        self._stop_flag = threading.Event()

    def run(self, pipeline: str = "__default__", env: str = None) -> Generator[str, None, None]:
        """Run kedro pipeline and yield output lines.

        Args:
            pipeline: Name of the pipeline to run. Defaults to "__default__" (full pipeline).
            env: Environment/config preset to use (e.g., 'quick_test', 'production').

        Yields:
            Output lines from the pipeline execution.
        """
        self._stop_flag.clear()

        cmd = ["kedro", "run"]
        if pipeline != "__default__":
            cmd.extend(["--pipeline", pipeline])
        if env:
            cmd.extend(["--env", env])

        # Set up environment for proper encoding on Windows
        env_vars = os.environ.copy()
        env_vars["PYTHONIOENCODING"] = "utf-8"

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=env_vars,
            )

            # Stream output line by line
            for line in iter(self.process.stdout.readline, ''):
                if self._stop_flag.is_set():
                    yield "[Pipeline cancelled by user]"
                    break
                if line:
                    yield line.rstrip()

            self.process.wait()

            if self._stop_flag.is_set():
                yield "[Pipeline cancelled]"
            else:
                exit_code = self.process.returncode
                if exit_code == 0:
                    yield "\n[Pipeline completed successfully]"
                else:
                    yield f"\n[Pipeline failed with exit code {exit_code}]"

        except FileNotFoundError:
            yield "[Error: kedro command not found. Is Kedro installed?]"
        except Exception as e:
            yield f"[Error running pipeline: {str(e)}]"
        finally:
            self.process = None

    def cancel(self) -> str:
        """Cancel the running pipeline.

        Returns:
            Status message.
        """
        self._stop_flag.set()
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            return "Pipeline cancelled"
        return "No pipeline running"

    def is_running(self) -> bool:
        """Check if pipeline is currently running.

        Returns:
            True if a pipeline is running, False otherwise.
        """
        return self.process is not None and self.process.poll() is None

    def get_available_pipelines(self) -> list:
        """Get list of available pipelines.

        Returns:
            List of pipeline names.
        """
        return [
            "__default__",
            "data",
            "prepare",
            "training",
            "inference",
            "evaluation",
            "backtest",
            "evaluate_only",
        ]
