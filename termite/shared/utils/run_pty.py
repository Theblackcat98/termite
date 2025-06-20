import os
import pty
import sys
import errno
from select import select, error as SelectError
from subprocess import Popen, TimeoutExpired

try:
    from termite.shared.utils.python_exe import get_python_executable
except ImportError:
    from python_exe import get_python_executable


def run_pty(command: str):
    python_exe = get_python_executable()
    masters, slaves = zip(pty.openpty(), pty.openpty())
    proc = Popen(
        [python_exe, command],
        stdin=slaves[0],
        stdout=slaves[0],
        stderr=slaves[1],
    )

    try:
        for fd in slaves:
            os.close(fd)

        readable = {
            masters[0]: sys.stdout.buffer,
            masters[1]: sys.stderr.buffer,
        }

        # Keep reading until EOF from both stdout/stderr or the process ends
        while True:
            if not readable:
                break

            try:
                rlist, _, _ = select(readable, [], [])
            except SelectError:
                break

            for fd in rlist:
                try:
                    data = os.read(fd, 1024)
                except OSError as e:
                    if e.errno != errno.EIO:
                        raise
                    del readable[fd]
                else:
                    if not data:
                        del readable[fd]
                    else:
                        readable[fd].write(data)
                        readable[fd].flush()

    finally:
        if proc.poll() is None:  # Check if process is still running
            proc.terminate()  # Send SIGTERM
            try:
                proc.wait(timeout=0.5)  # Short wait for graceful termination
            except TimeoutExpired:
                # SIGTERM failed or timed out, escalate to SIGKILL
                proc.kill()
                try:
                    proc.wait(timeout=0.5) # Wait for SIGKILL to take effect
                except TimeoutExpired:
                    # This would be unusual, means SIGKILL also failed
                    print(f"Error: Process {proc.pid} failed to terminate even after SIGKILL.", file=sys.stderr)
            except Exception as e:
                # Other exceptions during wait (e.g. InterruptedError)
                print(f"Error waiting for process {proc.pid} termination: {e}", file=sys.stderr)
                proc.kill() # Ensure kill if wait fails for other reasons

        # Ensure master file descriptors are closed
        for fd in masters:
            try:
                os.close(fd)
            except OSError:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    command = sys.argv[1]
    run_pty(command)
