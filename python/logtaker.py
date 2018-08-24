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


class Logtaker(object):
    """ Handling logging and error messages.

    This class allows to log all the information that is written to the
    screen to a file as well.
    """

    def __init__(self):
        self._error_log = []
        self.verbose = 1
        self.logfile = None
        self._nlend = True
        self._welcome_message_printed = False

    def error_message(self, msg, *args, **kwargs):
        """ report error and keep track of the error message

        Parameters
        ==========
        msg : str
            the error message
        """
        self._error_log += [msg.format(*args, **kwargs)]
        self.logged_message('ERROR: ' + msg, *args, **kwargs)

    def log_time(self):
        """ print the current time """
        self.logged_message(str(datetime.datetime.now()))

    def get_error_messages(self):
        """ get a list of all error messages raised """
        return self._error_log

    def clear_error_messages(self):
        """ clear the list of all error messages raised """
        del self._error_log[:]

    def welcome_message(self, always=False):
        """ print a welcome message """
        if always or not self._welcome_message_printed:
            self.log_time()
            self.logged_message("MaxEnt run")
            self.logged_message("TRIQS application maxent")
            self.logged_message(
                """Copyright(C) 2018 Gernot J. Kraberger
                Copyright (C) 2018 Simons Foundation
                Authors: Gernot J. Kraberger and Manuel Zingl""")
            self.logged_message('''This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions; see file LICENSE''')
            self.logged_message(
                'Please cite this code and the appropriate original papers (see documentation).\n')
            self._welcome_message_printed = True

    def verbosity_message(self, msg, iserr=False, *args, **kwargs):
        """ report message dependent on verbosity level

        Parameters
        ==========
        msg : str
            the message
        iserr : bool
            whether it has error character (True) or info character (False)
        """
        if iserr:
            self._verbosity_error_message(msg, *args, **kwargs)
        else:
            if self.verbose == 1:
                print('\r', msg.format(*args, **kwargs), end='')
                self._nlend = False
            elif self.verbose == 2:
                print(msg.format(*args, **kwargs))

    def _verbosity_error_message(self, msg, *args, **kwargs):
        if self.verbose == 1:
            print("\n\033[2K       " +
                  msg.format(*args, **kwargs) +
                  "\r\033[1A", end='')
            self._nlend = False
        elif self.verbose == 2:
            print(msg.format(*args, **kwargs))

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

    def logged_message(self, msg, *args, **kwargs):
        """ print a message and write it to the log file

        Parameters
        ==========
        msg : str
            the message
        """
        if not self._nlend:
            print()
        print(msg.format(*args, **kwargs))
        self._nlend = True
        if self.logfile is not None:
            self.logfile.write(msg.format(*args, **kwargs) + '\n')
