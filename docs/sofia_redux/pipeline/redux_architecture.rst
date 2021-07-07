Redux is designed to be a light-weight interface to data reduction
pipelines.  It contains the definitions of how reduction algorithms
should be called for any given instrument, mode, or pipeline,
in either a command-line interface (CLI) or graphical user
interface (GUI) mode, but it does not contain the reduction
algorithms themselves.

Redux is organized around the principle that
any data reduction procedure can be accomplished by running a linear
sequence of data reduction steps. It relies on a `Reduction` class that
defines what these steps are and in which order they should be run
(the reduction "recipe").  Reductions have an associated `Parameter`
class that defines what parameters the steps may accept.  Because
reduction classes share common interaction methods, they can be
instantiated and called from a completely generic front-end GUI,
which provides the capability to load in raw data files, and then:

#. set the parameters for a reduction step,
#. run the step on all input data,
#. display the results of the processing,

and repeat this process for every step in sequence to complete the
reduction on the loaded data. In order to choose the correct
reduction object for a given data set, the interface uses a `Chooser`
class, which reads header information from loaded input files and
uses it to decide which reduction object to instantiate and return.

The GUI is a PyQt  application, based around the `Application`
class. Because the GUI operations are completely separate from the
reduction operations, the automatic pipeline script is simply a wrapper
around a reduction object: the `Pipe` class uses the Chooser to
instantiate the Reduction, then calls its *reduce* method, which calls
each reduction step in order and reports any output files generated.
Both the Application and Pipe classes inherit from a common `Interface`
class that holds reduction objects and defines the methods for
interacting with them.  The `Application` class additionally may
start and update custom data viewers associated with the
data reduction; these should inherit from the Redux `Viewer` class.

All reduction classes inherit from the generic `Reduction` class,
which defines the common interface for all reductions: how parameters
are initialized and modified, how each step is called.
Each specific reduction class must then define
each data reduction step as a method that calls the appropriate
algorithm.

The reduction methods may contain any code necessary to
accomplish the data reduction step. Typically, a reduction method will
contain code to fetch the parameters for the method from the object’s
associated Parameters class, then will
call an external data reduction algorithm with appropriate parameter values,
and store the results in the 'input' attribute to be available for the
next processing step. If processing results in data that can be displayed,
it should be placed in the 'display_data' attribute, in a format that
can be recognized by the associated Viewers.  The Redux GUI checks
this attribute at the end of each data reduction step and displays the
contents via the Viewer's 'display' method.

Parameters for data reduction are stored as a list of `ParameterSet`
objects, one for each reduction step.  Parameter sets contain the key,
value, data type, and widget type information for every parameter.
A `Parameters` class may generate these parameter sets by
defining a default dictionary that associates step names with parameter
lists that define these values.  This dictionary may be defined directly
in the Parameters class, or may be read in from an external configuration
file or software package, as appropriate for the reduction.
