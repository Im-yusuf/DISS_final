from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_lead_reduction.data.preprocess import main


if __name__ == "__main__":
    main()