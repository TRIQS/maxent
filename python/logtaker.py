# TRIQS application maxent
# Copyright (C) 2018 Gernot J. Kraberger
# Copyright (C) 2018 Simons Foundation
# Authors: Gernot J. Kraberger and Manuel Zingl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import absolute_import, print_function
import datetime
from warnings import warn


class VerbosityFlags(object):
    """ Setting verbosity

    This describes different verbosity levels for controlling how much
    output is generated.

    Note that the actual integer values are considered implementation
    details and might not be conserved across versions.

    VerbosityFlags can be combined using ``|``, e.g. in order to write
    out header and timing information, use::

        self.verbose = VerbosityFlags.Header | VerbosityFlags.Timing

    Additional VerbosityFlags can be set with ``|=``::

        self.verbose |= VerbosityFlags.AlphaLoop

    and unset with ``&= ~``::

        self.verbose &= ~VerbosityFlags.AlphaLoop

    Testing for a particular flag can be done with ``&``.

    Use this in the :py:class:`.Logtaker` or in the
    :py:meth:`.MaxEntLoop.set_verbosity` method.

    The following shows a typical output of the code and together with
    the VerbosityFlags for the individual lines:

    .. code-block:: none

        ElementInfo     Calculating diagonal elements.
                        Calling MaxEnt for element 0 0
        Header          2018-11-22 16:51:11.012704
                        MaxEnt run
                        TRIQS application maxent
                        ...
                        scaling alpha by a factor 201 (number of data points)
        AlphaLoop       alpha[0] =   1.00500000e+05, chi2 =   6.90348835e+04, n_iter=       9
        SolverDetails   6 Q: 7.031977e+06, max_f: 1.494887e-05, conv: 5.297643e-16
        AlphaLoop       alpha[1] =   2.69610927e+04, chi2 =   2.69219648e+04, n_iter=       7
                        ...
        Timing          MaxEnt loop finished in 0:00:00.548325
    """

    Quiet = 0
    Header = 1
    ElementInfo = 2
    Timing = 4
    AlphaLoop = 8
    SolverDetails = 16
    Errors = 32

    Default = Header | ElementInfo | Timing | AlphaLoop | Errors


class Logtaker(object):
    """ Handling logging and error messages.

    This class allows to log all the information that is written to the
    screen to a file as well.
    """

    def __init__(self):
        self._error_log = []
        self.verbose = VerbosityFlags.Default

        # which verbosity messages should be printed in one line
        self.one_line = VerbosityFlags.SolverDetails

        self.logfile = None
        # the verbose level of the logfile; if None, use self.verbose
        self.logfile_verbose = None

        # the last line ending written
        self.end = '\n'
        self._welcome_message_printed = False

    def error_message(self, msg, *args, **kwargs):
        """ report error and keep track of the error message

        Parameters
        ==========
        msg : str
            the error message
        """
        self._error_log += [msg.format(*args, **kwargs)]
        self.message(VerbosityFlags.Errors, 'ERROR: ' + msg, *args, **kwargs)

    def log_time(self, message_verbosity=VerbosityFlags.Header):
        """ print the current time """
        self.message(message_verbosity, str(datetime.datetime.now()))

    def get_error_messages(self):
        """ get a list of all error messages raised """
        return self._error_log

    def clear_error_messages(self):
        """ clear the list of all error messages raised """
        del self._error_log[:]

    def welcome_message(self, always=False,
                        message_verbosity=VerbosityFlags.Header):
        """ print a welcome message """
        if always or not self._welcome_message_printed:
            self.log_time(message_verbosity=message_verbosity)
            self.message(message_verbosity, "MaxEnt run")
            self.message(message_verbosity, "TRIQS application maxent")
            self.message(message_verbosity,
                         "Copyright(C) 2018 Gernot J. Kraberger\n" +
                         "Copyright(C) 2018 Simons Foundation\n" +
                         "Authors: Gernot J. Kraberger and Manuel Zingl")
            self.message(
                message_verbosity,
                "This program comes with ABSOLUTELY NO WARRANTY.\n" +
                "This is free software, and you are welcome to redistribute" +
                "it under certain conditions; see file LICENSE.")
            self.message(
                message_verbosity,
                "Please cite this code and the appropriate original papers (see documentation).\n")
            self._welcome_message_printed = True

    def verbosity_message(self, msg, iserr=False, *args, **kwargs):
        raise NotImplementedError('The function verbosity_message was '
                                  'removed. Please use message instead.')

    def open_logfile(self, name, append=True):
        """ open the log file

        Parameters
        ==========
        name : str
            the name of the log file
        """
        self.logfile = open(name, 'a' if append else 'w')

    def close_logfile(self):
        """ close the log file """
        if self.logfile is not None:
            self.logfile.close()
        self.logfile = None

    def message(self, message_verbosity, msg, *args, **kwargs):
        """ print a message and write it to the log file

        Parameters
        ==========
        message_verbosity : :py:class:`.VerbosityFlags`
            verbosity flags that must be set in order to write that
            message (can be one flag or a ``|`` combination)
        msg : str
            the message; it can contain {} that are formatted with
            ``format`` using the ``args`` and ``kwargs``
        """

        # check whether all flags given in message_verbosity are set in
        # self.verbose
        if self.verbose & message_verbosity == message_verbosity:

            # if one of the verbosity flags given for this message
            # triggers one line behavior
            if message_verbosity & self.one_line:
                self.end = ''
                # go back to the start of the line
                print('\r', end=self.end)
            elif not self.end == '\n':
                # last printed line did not end with newline; print
                # newline now
                self.end = '\n'
                print('', end=self.end)

            print(msg.format(*args, **kwargs), end=self.end)

        if self.logfile is not None:
            logfile_verbose = self.logfile_verbose
            if logfile_verbose is None:
                logfile_verbose = self.verbose

            # there is no one line behavior for the logfile

            if logfile_verbose & message_verbosity == message_verbosity:
                self.logfile.write(msg.format(*args, **kwargs) + '\n')

    def logged_message(self, msg, *args, **kwargs):
        """ print a message and write it to the log file

        .. warning::

            The use of ``logged_message`` is deprecated.
            Use ``message`` instead.

        Parameters
        ==========
        msg : str
            the message
        """

        warnings.warn(
            "logged_message is deprecated. Use message instead.",
            DeprecationWarning)

        self.message(0, msg, *args, **kwargs)

    def solver_verbose_callback(self, *args, **kwargs):
        self.message(VerbosityFlags.SolverDetails, *args, **kwargs)
