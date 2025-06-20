import unittest
import os
import sys
from pathlib import Path
import psutil
import time

# Adjust path to import from termite project root
current_dir = Path(__file__).parent
project_root = current_dir.parent # termite/tests/ -> termite/
# If termite is in a src directory, this might need adjustment, e.g. current_dir.parent.parent
# Assuming standard project structure where 'termite' package is at project_root
sys.path.insert(0, str(project_root.parent)) # Add parent of 'termite' package dir to path

from termite.dtos import Script # This will now try to import from project_root/termite/dtos
from termite.shared.run_tui import run_in_pseudo_terminal
# from termite.shared.utils.python_exe import get_python_executable # Will be used in next step

MINIMAL_TUI_SCRIPT_NAME = "test_minimal_tui.py"
MINIMAL_TUI_SCRIPT_PATH = current_dir / MINIMAL_TUI_SCRIPT_NAME

class TestProcessLeaks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create the minimal TUI script if it doesn't exist (it should be created by this subtask)
        # This is more of a check or fallback. The file is created as part of the subtask.
        if not MINIMAL_TUI_SCRIPT_PATH.exists():
            script_content = """\
import urwid

def main():
    txt = urwid.Text(u"Minimal TUI for testing. Auto-exiting...")
    fill = urwid.Filler(txt, 'top')
    loop = urwid.MainLoop(fill)
    loop.set_alarm_in(0.2, lambda _loop, _data: _loop.stop()) # Auto-exit
    loop.run()

if __name__ == '__main__':
    main()
"""
            with open(MINIMAL_TUI_SCRIPT_PATH, "w") as f:
                f.write(script_content)

    @classmethod
    def tearDownClass(cls):
        # Clean up the minimal TUI script (optional, could leave it for inspection)
        # For CI, it's good to clean up.
        # if os.path.exists(MINIMAL_TUI_SCRIPT_PATH):
        #     os.remove(MINIMAL_TUI_SCRIPT_PATH)
        pass # Let's leave it for now.

    def get_python_processes_spawned_by_current_process(self):
        current_process = psutil.Process()
        # On some systems, children are not cleaned up immediately if the parent (this test process)
        # doesn't explicitly wait for them, or if they become daemons.
        # Popen in run_pty should be waited on, so direct children of run_pty should be gone.
        # We are looking for children of *this* test process.
        children = current_process.children(recursive=True)

        python_children = []
        for proc in children:
            try:
                # Check process name (e.g., 'python', 'python3')
                # proc.exe() would be more specific but might vary more (e.g. /usr/bin/python3.8)
                if "python" in proc.name().lower() and proc.is_running():
                    python_children.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied): # Process might terminate during check
                continue
        return python_children

    def test_run_tui_repeatedly_for_leaks(self):
        num_iterations = 3 # Keep it low for CI, can be increased for local deep testing
        if not MINIMAL_TUI_SCRIPT_PATH.exists():
            self.fail(f"Minimal TUI script not found at {MINIMAL_TUI_SCRIPT_PATH}")

        script_code_content = MINIMAL_TUI_SCRIPT_PATH.read_text()

        initial_python_children_pids = {p.pid for p in self.get_python_processes_spawned_by_current_process()}
        # print(f"Initial Python children PIDs: {initial_python_children_pids}") # For debugging

        potentially_leaked_procs_details = []

        for i in range(num_iterations):
            # print(f"Test iteration {i+1}/{num_iterations}")

            # Get child processes right before the run call for this iteration
            # This helps isolate leaks specific to this iteration if initial state is complex
            children_pids_before_this_run = {p.pid for p in self.get_python_processes_spawned_by_current_process()}

            current_script_obj = Script(code=script_code_content)
            # The timeout for run_in_pseudo_terminal is for the script execution itself.
            stdout, stderr = run_in_pseudo_terminal(current_script_obj, timeout=3)

            # Basic script functional checks
            if "Traceback" in stderr:
                self.fail(f"Script stderr had a traceback on iteration {i+1}: {stderr}")
            if "Error" in stderr and "KeyboardInterrupt" not in stderr : # KeyboardInterrupt can happen on timeout
                 self.fail(f"Script stderr had an error on iteration {i+1}: {stderr}")

            # Crucial: Allow a brief moment for all cleanup in run_pty's finally block to complete.
            time.sleep(0.5) # Adjusted based on typical process cleanup speed.

            current_python_children = self.get_python_processes_spawned_by_current_process()

            for proc in current_python_children:
                # A process is suspicious if:
                # 1. It wasn't in the initial set of all python children (it's new since test started)
                # AND
                # 2. It wasn't present just before this specific iteration's run_in_pseudo_terminal call
                #    (or if we want to catch any accumulation, compare against initial_python_children_pids)
                # Let's consider any python child not in the initial set as potentially problematic if it persists.
                if proc.pid not in initial_python_children_pids:
                    try:
                        cmd_line = " ".join(proc.cmdline()) if proc.cmdline() else "N/A"
                        detail = f"PID: {proc.pid}, Name: {proc.name()}, Cmd: {cmd_line}"
                        if detail not in potentially_leaked_procs_details: # Avoid duplicates if checked multiple times
                           potentially_leaked_procs_details.append(detail)
                        # print(f"Suspicious Python process found on iter {i+1}: {detail}") # For debugging
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            # If we want to be very strict and ensure *no new* python processes stick around *after each iteration*
            # compared to state *before that iteration*:
            # new_persistent_children_this_iter = []
            # for proc in current_python_children:
            #    if proc.pid not in children_pids_before_this_run:
            #        # This process is new since before this iteration's call
            #        # If it's still here, it's a leak from this iteration
            #        cmd_line = " ".join(proc.cmdline()) if proc.cmdline() else "N/A"
            #        new_persistent_children_this_iter.append(f"PID: {proc.pid}, Cmd: {cmd_line}")
            # self.assertEqual(len(new_persistent_children_this_iter), 0,
            #    f"Iteration {i+1} leaked processes: {new_persistent_children_this_iter}")


        # The final check: are there any new python processes compared to the very start of the test?
        self.assertEqual(len(potentially_leaked_procs_details), 0,
                         f"Found {len(potentially_leaked_procs_details)} new and persistent Python child processes after {num_iterations} iterations: {potentially_leaked_procs_details}")

if __name__ == "__main__":
    # This allows running the test file directly.
    # For subtask, it might try to run this, ensure it can find modules.
    unittest.main()
