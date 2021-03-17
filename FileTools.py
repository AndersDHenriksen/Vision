import shutil
from distutils.dir_util import copy_tree
from pathlib import Path
import urllib.request
import zipfile
import tarfile


def count_files(parent_folder, file_string, recursive=False):
    """
    Get the number of files which matches the file_string, * allowed. Also useful to check file existence.
    """
    if recursive:
        return len(list(Path(parent_folder).rglob(file_string)))
    else:
        return len(list(Path(parent_folder).glob(file_string)))


def limit_n_files(parent_folder, file_string, n_files):
    """
    Limit the number of a specific files inside a folder. Files are deleted in ascending order until n_files are left.
    """
    file_paths = sorted(list(Path(parent_folder).glob(file_string)))
    [fp.unlink() for fp in file_paths[:-n_files]]


def download_url(url, output_path=None, do_unzip=True):
    """
    Download a file from an URL.
    :param url: URL of file to download
    :type url: str
    :param output_path: Path to place downloaded file
    :type output_path: Union[str, pathlib.Path]
    :param do_unzip: Whether to unzip if file is an archive
    :type do_unzip: bool
    :return: Output path
    :rtype: Union[str, pathlib.Path]
    """
    description = url.split('/')[-1]
    output_path = output_path or description
    try:
        from tqdm import tqdm

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    except ImportError:
        urllib.request.urlretrieve(url, filename=output_path)
    if do_unzip:
        output_path = extract_file(output_path, delete_archive=True, fail_silently=True)
    return output_path


def extract_file(compressed_file_path, delete_archive=False, fail_silently=False):
    """
    Try to extract a zip/tar file. Output will be placed next to zip/tar file.
    :param compressed_file_path: Path to zip/tar file.
    :type compressed_file_path: Union[str, pathlib.Path]
    :param delete_archive: Whether to delete archive after extraction
    :type delete_archive: bool
    :param fail_silently: Allow silent failure
    :type fail_silently: bool
    :return: Path to decompressed folder
    :rtype: pathlib.Path
    """
    file_path = Path(compressed_file_path)
    out_folder = file_path.parent
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(out_folder)
    elif tarfile.is_tarfile(file_path):
        with tarfile.TarFile(file_path, 'r') as tar_ref:
            tar_ref.extractall(out_folder)
    else:
        if fail_silently:
            return file_path
        else:
            raise Exception("Cannot handle compression format")
    if delete_archive:
        file_path.unlink()
    return file_path.parent / file_path.stem


def download_h264_codec():
    """
    Download h264 codec file to current directory.
    """
    if count_files(Path.cwd(), 'openh264*.dll'):
        return
    url = r"https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2"
    bz2_file = download_url(url)
    import bz2
    with bz2.BZ2File(bz2_file) as zipfile:  # open the file
        data = zipfile.read()  # get the decompressed data
    newfilepath = str(bz2_file)[:-4]  # assuming the filepath ends with .bz2
    open(newfilepath, 'wb').write(data)
    bz2_file.unlink()


def rmtree(target):
    """ Alias for shutil.rmtree. Note Path.rmdir can only delete empty. """
    assert Path(target).is_dir()
    shutil.rmtree(target)


# Create monkey patches
def _copy(self, target):
    """ Monkey patch for shutil.copy. """
    Path(target).parent.mkdir(exist_ok=True)
    return shutil.copy(self, target) if self.is_file() else copy_tree(str(self), str(target))


def _move_to_subfolder(self, subfolder_name):
    """ Move file to subfolder. Subfolder will be created if non-existing. """
    new_path = self.parent / subfolder_name / self.name
    new_path.parent.mkdir(exist_ok=True)
    return self.rename(new_path)


# Apply monkey patches
Path.copy = _copy
Path.move_to_subfolder = _move_to_subfolder
