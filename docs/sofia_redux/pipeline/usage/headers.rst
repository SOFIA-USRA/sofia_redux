From the Redux interface, the **Display Settings** can be used to:

- Set the FITS extension to display (**First**, or edit to enter
  a specific extension), or specify that all extensions should
  be displayed in a cube or in separate frames.
- Lock individual frames together, in image or WCS coordinates.
- Lock cube slices for separate frames together, in image or
  WCS coordinates.
- Set the image scaling scheme.
- Set a default color map.
- Zoom to fit image after loading.
- Tile image frames, rather than displaying a single frame at a
  time.

Changing any of these options in the Data View tab will cause the
currently displayed data to be reloaded, with the new options.
Clicking **Reset Display Settings** will revert any edited options
to the last saved values.  Clicking **Restore Default Display Settings**
will revert all options to their default values.

In the **QAD Tools** section of the **Data View** tab, there are
several additional tools available.

Clicking the **ImExam** button
(scissors icon) launches an event loop in DS9.  After launching it,
bring the DS9 window forward, then use the keyboard to perform interactive
analysis tasks:

- Type 'a' over a source in the image to perform photometry at the
  cursor location.
- Type 'p' to plot a pixel-to-pixel comparison of all frames at the
  cursor location.
- Type 's' to compute statistics and plot a histogram of the data
  at the cursor location.
- Type 'c' to clear any previous photometry results or active plots.
- Type 'h' to print a help message.
- Type 'q' to quit the ImExam loop.

The photometry settings (the image window considered, the model fit,
the aperture sizes, etc.) may be customized in the **Photometry Settings**.
Plot settings (analysis window size, shared plot axes, etc.) may be
customized in the **Plot Settings**.
After modifying these settings, they will take effect only for new
apertures or plots (use 'c' to clear old ones first).  As for the display
settings, the reset button will revert to the last saved values
and the restore button will revert to default values.
For the pixel-to-pixel and histogram plots, if the cursor is contained within
a previously defined DS9 region (and the `regions` package is installed),
the plot will consider only pixels within the region.  Otherwise, a window
around the cursor is used to generate the plot data.  Setting the window
to a blank value in the plot settings will use the entire image.

Clicking the **Header** button (magnifying glass icon) from the
**QAD Tools** section opens a new window that displays headers
from currently loaded FITS files in text form (|ref_headers|).
The extensions displayed depends on the extension
setting selected (in **Extension to Display**).  If a particular extension is
selected, only that header will be displayed.  If all extensions
are selected (either for cube or multi-frame display), all extension
headers will be displayed.  The buttons at the bottom of the window
may be used to find or filter the header text, or generate a table
of header keywords.  For filter or table display, a comma-separated
list of keys may be entered in the text box.

Clicking the **Save Current Settings** button (disk icon) from the
**QAD Tools** section saves all current display and photometry
settings for the current user.  This allows the user's settings to
persist across new Redux reductions, and to be loaded when Redux
next starts up.
