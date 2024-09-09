from os import path, makedirs

from abc import ABC, abstractmethod

from datetime import datetime

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class MissingMandatorySavingInfo(Exception):
    pass


class MissingFilenameError(Exception):
    pass


class Saver(ABC):
    """Abstract base class for saving objects with mandatory and optional saving information.

    Args:
        object_to_save: The object to be saved.
        filename (str): The name of the file to save the object to.
        saving_info (dict): Dictionary containing information about saving directories, namespaces, and other options.

    Attributes:
        saving_info (dict): Stores both mandatory and optional fields related to saving.
        object_to_save: The object that needs to be saved.
        save_dir (str): The directory path where the file will be saved.
        filename (str): The formatted filename for saving the object.
    """

    def __init__(self, object_to_save, filename, saving_info):
        """Initializes the Saver class with the object, filename, and saving information.

        Raises:
            MissingFilenameError: If no filename is provided when `save=True`.
        """
        if filename == "":
            raise MissingFilenameError(
                "You have enabled save = True, but no filename has been specified. "
                "Please specify the filename of the object you want to save"
            )

        super().__init__()
        self.saving_info = {}
        self.object_to_save = object_to_save

        self.populate_mandatory_fields(saving_info)
        self.populate_optional_fields(saving_info)
        self.save_dir = self._get_save_dir(
            self.saving_info["top_dir"], self.saving_info.get("deep_dir")
        )
        self.check_if_dir_exists()
        self.filename = "_".join((*self.saving_info["namespace"], filename))

    def __call__(self):
        """Calls the `save` method when the instance is invoked."""
        self.save()

    @abstractmethod
    def save(self):
        """Abstract method to be implemented by subclasses for saving the object."""
        pass

    @abstractmethod
    def raise_missing_mandatory_field(self, key):
        """Abstract method for raising errors when mandatory fields are missing."""
        pass

    @abstractmethod
    def log_missing_optional_field(self, key):
        """Abstract method for logging missing optional fields."""
        pass

    @property
    def mandatory_keys(self) -> list:
        """List of mandatory keys required for saving."""
        return ["top_dir", "namespace"]

    @property
    def optional_keys(self) -> list:
        """List of optional keys that can be provided for saving."""
        return ["deep_dir", "extensions"]

    def populate_mandatory_fields(self, saving_info: dict):
        """Populates the mandatory fields from the saving information.

        Args:
            saving_info (dict): The dictionary containing saving information.

        Raises:
            The specific method `raise_missing_mandatory_field` if a mandatory field is missing.
        """
        for mk in self.mandatory_keys:
            try:
                self.saving_info[mk] = saving_info[mk]
            except KeyError:
                self.raise_missing_mandatory_field(mk)

    def populate_optional_fields(self, saving_info: dict):
        """Populates the optional fields from the saving information.

        Args:
            saving_info (dict): The dictionary containing saving information.

        Logs missing optional fields using the `log_missing_optional_field` method.
        """
        for ok in self.optional_keys:
            try:
                self.saving_info[ok] = saving_info[ok]
            except KeyError:
                self.saving_info[ok] = None
                self.log_missing_optional_field(ok)

    def check_if_dir_exists(self):
        """Checks if the directory exists; if not, creates the directory."""
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)

    @staticmethod
    def _get_save_dir(top_dir, deep_dir):
        """Generates the save directory path.

        Args:
            top_dir (str): The top-level directory.
            deep_dir (str, optional): The subdirectory to organize files, based on date.

        Returns:
            str: The complete directory path for saving.
        """
        if deep_dir:
            today = datetime.today().strftime("%Y-%m-%d")
            dir_name = today if deep_dir is None else "-".join((today, deep_dir))
            outdir = path.join(top_dir, dir_name)
        else:
            outdir = top_dir

        return outdir


class PlotSaver(Saver):
    """Saves plot objects (e.g., figures) with additional handling for file extensions and logging.

    Inherits from the `Saver` class and provides specific functionality for saving plot objects.

    Args:
        object_to_save: The plot object (e.g., Matplotlib figure) to be saved.
        filename (str): The name of the file to save the plot to.
        saving_info (dict): Dictionary containing information on directories, namespaces, and file extensions.

    Attributes:
        saving_info (dict): Contains both mandatory and optional fields related to saving.
    """

    def __init__(self, object_to_save, filename, saving_info):
        """Initializes the PlotSaver with the object, filename, and saving information."""
        super().__init__(object_to_save, filename, saving_info)

    @staticmethod
    def get_key_description(key: str) -> str:
        """Provides a description of the specified saving info key.

        Args:
            key (str): The key for which the description is requested.

        Returns:
            str: The description of the key.

        Example:
            >>> PlotSaver.get_key_description('top_dir')
            'The top directory that your objects should be saved in'
        """
        key_descriptions = {
            "top_dir": "The top directory that your objects should be saved in",
            "namespace": "A list of strings that build the name of the object. These are by default set internally by SysVar",
            "deep_dir": "A deeper directory inside the top_dir that specifies the final location of the saved file",
            "extensions": "Extra extensions for the file to be saved. For figures this defaults to pdf and png",
        }
        return key_descriptions[key]

    def raise_missing_mandatory_field(self, key: str):
        """Raises an error if a mandatory field is missing.

        Args:
            key (str): The missing mandatory field key.

        Raises:
            MissingMandatorySavingInfo: If the mandatory field is not found in `saving_info`.
        """
        raise MissingMandatorySavingInfo(
            f"\n You are attempting to save a Figure but you're missing the mandatory field {key}. \n"
            f"SysVar will not save objects in default locations, or with default names. \n"
            f"Please call the register_figure_saving_info method in order to specify the necessary information. \n"
            f"A description of {key} follows in the next line: \n {self.get_key_description(key)}"
        )

    def log_missing_optional_field(self, key: str):
        """Logs a warning if an optional field is missing.

        Args:
            key (str): The missing optional field key.
        """
        logging.warn(
            f"\n You are attempting to save a Figure but you're missing the optional field {key}. \n"
            f"This will not prevent SysVar from saving the Figure, but be aware that you could have more control over your saved object. \n"
            f"A description of {key} follows in the next line: \n {self.get_key_description(key)}"
        )

    def add_default_extensions(self):
        """Ensures default file extensions (png and pdf) are added if none are provided."""
        if self.saving_info["extensions"] is None:
            self.saving_info["extensions"] = ["png", "pdf"]
        else:
            if "png" not in self.saving_info["extensions"]:
                self.saving_info["extensions"].append("png")
            if "pdf" not in self.saving_info["extensions"]:
                self.saving_info["extensions"].append("pdf")

    def save(self):
        """Saves the plot object in the specified directory with the given extensions.

        Ensures default extensions are added, then loops over the extensions to save the figure.

        Logs the location where the figure is saved.
        """
        self.add_default_extensions()

        # Loop over the extensions, create the filename and save the figure
        for ext in self.saving_info["extensions"]:
            # Ensure the extension has a dot prefix
            if ext[0] != ".":
                ext = "." + ext

            figname = self.filename + ext

            # Save the figure using the specified file name and directory
            self.object_to_save.savefig(
                path.join(self.save_dir, figname), bbox_inches="tight", dpi=800
            )

        logging.info(f"Saved figures in {self.save_dir}")
