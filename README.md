# Cube Inspector v0.1.1

Simple GUI program for inspecting and analyzing hyperspectral image 
cubes that have been saved in ENVI format.

The UI elements do have tooltips, so it should be fairly easy to decipher 
what you can do with it. Clicking on the RGB image of the cube will plot its 
spectra and click-dragging will plot mean spectra of the dragged square. 

Cube Inspector tries to save your current state (image cubes and used RGB bands) 
upon closing.

Note that this is veeeeery early version of the software, so be prepared for 
crashes and weird behavior. There is an embedded console window to help you. 
No logs though at the moment.

# New stuff in v0.1.1

- Now you can set the range of spectrum plot to filter out bands you dont
  want to see
- Can select the bands used for RGB false color representation from the UI.
  You can also fill one or more RGB channels with an int by typing e.g. "f0" to
  fill with zero.

  
