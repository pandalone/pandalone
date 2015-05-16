#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, unicode_literals


__commit__ = ""


# From: http://stackoverflow.com/questions/2216173/how-to-get-path-of-start-menus-programs-directory
#
def win_shell():
    from win32com.client import Dispatch
    return Dispatch('WScript.Shell')


def win_folder(wshell, folder_name, folder_csidl=None):
    """

    :param wshell: win32com.client.Dispatch('WScript.Shell')
    :param str folder_name: ( StartMenu | MyDocuments | ... )
    :param str folder_csidl: see http://msdn.microsoft.com/en-us/library/windows/desktop/dd378457(v=vs.85).aspx
    """
    # from win32com.shell import shell, shellcon                          #@UnresolvedImport
    #folderid = operator.attrgetter(folder_csidl)(shellcon)
    #folder = shell.SHGetSpecialFolderPath(0, folderid)
    folder = wshell.SpecialFolders(folder_name)

    return folder

# See: http://stackoverflow.com/questions/17586599/python-create-shortcut-with-arguments
#    http://www.blog.pythonlibrary.org/2010/01/23/using-python-to-create-shortcuts/
#    but keep that for the future:
# forgot chose:
# http://timgolden.me.uk/python/win32_how_do_i/create-a-shortcut.html


def win_create_shortcut(wshell, path, target_path, wdir=None, target_args=None, icon_path=None, desc=None):
    """

    :param wshell: win32com.client.Dispatch('WScript.Shell')

    """

    is_url = path.lower().endswith('.url')
    shcut = wshell.CreateShortCut(path)
    try:
        shcut.Targetpath = target_path
        if icon_path:
            shcut.IconLocation = icon_path
        if desc:
            shcut.Description = desc
        if target_args:
            shcut.Arguments = target_args
        if wdir:
            shcut.WorkingDirectory = wdir
    finally:
        shcut.save()


if __name__ == '__main__':  # pragma: no cover
    raise NotImplementedError
