"""The entry point for Streamlit Community Cloud: the chatbot, as a hosted app.

Community Cloud runs one script and puts only that script's folder on the import
path -- here `ui/`, where it also finds `requirements.txt`. The apps import
`ui.*` and `src.*` from the repository root, so the root is put on the path
first, and then the chat app is run as if it were the script itself. It is run,
not imported: Streamlit reruns this file on every interaction, and an imported
module would only execute once.

Everything else is configured through the app's secrets; see
deploy/streamlit-cloud/DEPLOY.md.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "ui" / "chat_app.py"), run_name="__main__")
