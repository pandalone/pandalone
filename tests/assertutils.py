import os


class CustomAssertions(object):
    # Idea from: http://stackoverflow.com/a/15868615/548792

    def _raise_file_msg(self, msg, path, user_msg):
        msg = msg % os.path.abspath(path)
        if user_msg:
            msg = '%s: %s' % (msg, user_msg)
        raise AssertionError(msg)

    def assertFileExists(self, path, user_msg=None):
        if not os.path.lexists(path):
            self._raise_file_msg(
                "File does not exist in path '%s'!", path, user_msg)

    def assertFileNotExists(self, path, user_msg=None):
        if os.path.lexists(path):
            self._raise_file_msg("File exists in path '%s'!", path, user_msg)
