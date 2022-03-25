# MC19 Mask Fibre Analysis
https://aip.scitation.org/doi/10.1063/5.0074229

This is a Final Year Project (FYP) undertaken by Jake Wilkins at the University of Surrey Physics Department. 

This repository contains a number of files which process, analyse and output imagery relating to mask fibres. If you wish to run this code, input imagery is required, please contact me if you wish to do this.

Below is a basic description of what each of these files contains. 

## Folders
- POC are proof-of-concept files, these contain a number of proof of concepts for a number of processing methods that acted as a foothold for further work.
- Legacy contains a number of code which is no longer being used or is broken, i.e. not making up any part of the final report
- The base folder contains the rest of the files, all of which are written in python and most of which were used wholly or in part to create the final report

## Packages
Packages used in the main set of files are as follows:

- numpy
- cv2 - image manipulation
- pandas - data manipulation
- plotly - plotting results
- tifffile - outputting 3D results
- skimage (as part of scikit-image) - used for ingest and filters
- cc3d - connected components analysis in 3D space

## Files
Each file within this repository contains a summary at the top of the file written in a python comment. These cover the intended purpose of each file, but may not reflect everything the file does.

Development is regular, so a files outputs may change regularly.

**Key files to the report are:**

- processDataTo3D.py
- channelWidth2D.py
- findAirGaps.py
