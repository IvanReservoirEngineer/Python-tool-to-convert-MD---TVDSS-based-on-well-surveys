# Python-tool-to-convert-MD---TVDSS-based-on-well-surveys
Gewtting true vertical depth (TVDSS) when you only have measured depth (MD), and vice versa

# MD ↔ TVDSS converter for well trajectories

Small Python tool to convert between **measured depth (MD)** and **true vertical depth sub-sea (TVDSS)** using a well trajectory database.

## What it does

- Reads a text file with well trajectories in the format:

  `well   X   Y   MD   TVDSS   INCL   AZIM`

- Reads an Excel file with a list of wells and target depths (either MD or TVDSS).
- For each requested depth:
  - performs linear interpolation along the trajectory to get the corresponding MD or TVDSS,
  - if the target is outside the known trajectory, extrapolates along the last segment and marks these rows as `extrapolated`.

All examples in this repo use **synthetic/anonymised data**.

## Requirements

- Python 3.9+ (Anaconda / Miniconda is fine)
- Packages:
  - `pandas`
  - `numpy`

You can install them with:

```bash
pip install pandas numpy
