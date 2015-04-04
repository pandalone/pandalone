import os
import unittest


class CustomAssertions(object):
    # Idea from: http://stackoverflow.com/a/15868615/548792

    def _raise_file_msg(self, msg, path):
        raise AssertionError(msg % os.path.abspath(path))

    def assertFileExists(self, path):
        if not os.path.lexists(path):
            self._raise_file_msg("File does not exist in path '%s'!", path)

    def assertFileNotExists(self, path):
        if os.path.lexists(path):
            self._raise_file_msg("File exists in path '%s'!", path)
