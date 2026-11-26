"""Plugin hook launcher: runs lowtoken from the plugin folder, no install needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lowtoken.hook import main  # noqa: E402

sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "post"))
