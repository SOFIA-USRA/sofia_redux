The current set of parameters can be displayed, saved to a file,
or reset all at once using the **Parameters** menu. A previously
saved set of parameters can also be restored for use with the
current reduction (**Parameters -> Load Parameters**).

After all parameters for a step have been examined and set to the
user's satisfaction, a processing step can be run on all loaded
files either by clicking **Step**, or the **Run** button next to the
step name. Each processing step must be run in order, but if a
processing step is selected in the **Step through:** widget,
then clicking **Step** will treat all steps up through the selected
step as a single step and run them all at once. When a step has
been completed, its buttons will be grayed out and inaccessible.
It is possible to undo one previous step by clicking **Undo**.
All remaining steps can be run at once by clicking **Reduce**.
After each step, the results of the processing may be displayed
in a data viewer. After running a pipeline step or reduction,
click **Reset** to restore the reduction to the initial state,
without resetting parameter values.

Files can be added to the reduction set (**File -> Add Files**) or
removed from the reduction set (**File -> Remove Files**), but
either action will reset the reduction for all loaded files.
Select the **File Information** tab to display a table of information
about the currently loaded files (|ref_file_info|).
