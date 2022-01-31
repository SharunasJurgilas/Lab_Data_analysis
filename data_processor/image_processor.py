import numpy as np
import zipfile
import pandas as pd
import imageio
from utils.shape_estimator import fit_gauss_1d, fit_gauss_2d


def images_to_arrays(path, date, file_numbers, species):
        """
        Opens the folders file_numbers (list) from specified path (self.path). Each zipped folder contains .tiff
        image files which are labelled in ascending order, corresponding to the time they were acquired. This function
        preserves the ordering of files- required for matching data to variable experiment parameters.

        Parameters
        ----------
        path: folder name
        date: date of data folder. For example 01Jan21 stands for 1st of January 2021.
        file_numbers: a list containing the file numbers
        species: species being imaged. Set to 'CaF' or 'Rb'. This parameter is required to set the correct prefix
        for the image names. 'C' prefix is for CaF and 'R' prefix is for Rb.

        Returns a numpy array containing the image data. The array shape is (number of files, number of images in each
        file, number of pixels along y (rows), number of pixels along x (columns)).

        Automatically checks if cropping is required and returns a cropped image if it is.
        Automatically checks if image type is absorption and converts pixel intensities to optical densities if it is.
        """

        # Set prefix for image names depending on species being imaged:
        if species == 'Rb':
            prefix = 'R'
        elif species == 'CaF':
            prefix = 'C'

        folders = [path + '\\' + 'CaF' + date + '00' + '_' + format(number, '03') +
                   '.zip' for number in file_numbers]

        image_array_container = []
        for folder in folders:
            archive = zipfile.ZipFile(folder)
            images = [im for im in archive.namelist() if (prefix + 'Image' in im)]
            naturals = lambda names: [i[i.find("_") + 1:i.find(".")] for i in names]
            images = sorted(images, key=naturals)
            images_array = [imageio.imread(archive.read(im)) for im in images]
            archive.close()
            images_array = np.array(images_array, 'float64')
            image_array_container.append(images_array)
        return np.array(image_array_container, 'float64')


def get_parameters(path, date, txt_name, file_numbers, parameter):
    """
    Identical to images_to_arrays() but this time looks for the .txt files which contain experiment parameters.

    Parameters
    ----------
    path: folder name
    date: date of data folder. For example 01Jan21 stands for 1st of January 2021.
    txt_name: the name of the .txt file. Typically there are two. 'parameters' stores the analogue and digital
    pattern parameters. 'hardware' stores parameters used by the hardware controller (e.g. microwave synth).
    parameter: the name of the parameter which was varied in the experiment.

    Return the list of parameters in machine units. If a parameter corresponds to timing, 1 unit == 10 us.
    """
    folders = [path + '\\' + 'CaF' + date + '00' + '_' + format(number, '03') +
                '.zip' for number in file_numbers]
    parameter_container = []
    for folder in folders:
        archive = zipfile.ZipFile(folder)
        files = [x for x in archive.namelist() if x.endswith(txt_name + '.txt')]
        params = pd.read_csv(archive.open(files[0]), sep='\t', header=None, index_col=0)
        archive.close()
        param = float(params.loc[parameter, 1])
        parameter_container.append(param)
    return np.array(parameter_container, 'float64')


class ImageProcessor:
    """
    Class which defines various transformations on the images data set. The class is instantiated with the image
    and background data containers.

    The class contains methods to retrieve basic properties of the cloud: number of particles, size, position.

    The following methods should be applied to an instance of the class by the user to modify the attribute
    _image_container:
    crop_images(crop_center_x, crop_center_y, crop_size_x, crop_size_y) : crops images to specified dimensions;
    subtract_background() : subtracts a _backgroun_container from _image_container;
    average() : averages images in _image_container
    convert_to_od() : converts absorption images in _image_container to optical density maps
    remove_negative_pixels() : sets all negative pixel values in _image_container to 0.0
    subtract_constant_offset(offset) : subtracts a constant or array of constants from each pixel in _image_container.

    """

    pixel_size = 6.45 * 1e-3  # Camera single pixel size in mm
    t_exp = 10 * 1e-3  # Exposure time in s
    detuning = 14.5  # Absorption probe detuning in MHz
    fluorescence_rate = 1.5  # In MHz.
    collection_efficiency = 0.001  # Fluorescence imaging system total collection efficiency
    sigma_0 = 3 * (780 * 1e-6) ** 2 / 6.28  # Light absorption cross-section (2-level system)

    def __init__(self,
                 image_container=np.array([]),
                 background_container=np.array([]),
                 image_type='Fluorescence',
                 magnification=0.47,
                 pixel_binning=8):
        self._image_container = image_container
        self._background_container = background_container
        self.image_type = image_type
        self._magnification = magnification
        self._pixel_binning = pixel_binning

    def crop_images(self, crop_center_x, crop_center_y, crop_size_x, crop_size_y):
        """
        Crop images. Apply as a first transformation, if at all needed. The resulting array with cropped
        images will be stored as the image_container class attribute.

        Parameters
        ----------
        crop_center_x: recenter image along x
        crop_center_y: recenter image along y
        crop_size_x: total horizontal size after cropping
        crop_size_y: total vertical size after cropping
        """
        c_x1 = int(crop_center_y - crop_size_y / 2)
        c_x2 = int(crop_center_y + crop_size_y / 2)
        c_y1 = int(crop_center_x - crop_size_x / 2)
        c_y2 = int(crop_center_x + crop_size_x / 2)
        self._image_container = self._image_container[:, :, c_x1:c_x2, c_y1:c_y2]
        if self.image_type == 'Fluorescence':
            self._background_container = self._background_container[:, :, c_x1:c_x2, c_y1:c_y2]
        return self

    def subtract_background(self):
        """
        Subtract an averaged background from all images. Automatically checks image_type. If image_type == 'Absorption'
        raises ValueError because background absorption images are contained within the image container.
        """
        if self.image_type == 'Absorption':
            raise ValueError('Absorption image files contain background images. The convert_to_od() class method '
                             'subtracts the background automatically and updates the image container with the '
                             'processed images.')
        mean_background = self._background_container.mean(axis=1)
        self._image_container = self._image_container - mean_background
        return self

    def subtract_two_trigger_background(self):
        """
        Subtracts the background from images in two-trigger experiments. Updates image_container. First apply
        convert_to_two_trigger_data() method to transform image array into the correct shape.
        """
        if len(self.image_container) != 0 and type(self.image_container) != tuple:
            raise Exception('First convert image array to two trigger data. Use convert_to_two_trigger_data().')
        bg_1 = self._background_container[:, ::2, ...].mean(axis=1)
        bg_2 = self._background_container[:, 1::2, ...].mean(axis=1)
        self._image_container = self._image_container[0] - bg_1, self._image_container[1] - bg_2
        return self

    def convert_to_two_trigger_data(self):
        """
        Transform image_container to tuple containing normalization images and secondary images.
        """
        self._image_container = self._image_container[:, ::2, ...], self._image_container[:, 1::2, ...]
        return self

    def get_normalized_signal(self, averaged=True):
        """
        Returns the ratios (total signal in second image) / (total signal in first image).
        Parameters
        ----------
        averaged: default True- returns mean ratios and standard errors. If False returns raw ratios in array of
        shape (n experiments, n repetitions).
        """
        s = self._image_container[1].sum(axis=(-1, -2))
        n = self._image_container[0].sum(axis=(-1, -2))
        reps = s.shape[-1]
        if averaged:
            return (s / n).mean(axis=1), (s / n).std(axis=1) / np.sqrt(reps)
        else:
            return s / n

    def average(self):
        """
        Updates image_container to averaged images. Averages across the experiment repetition axis.
        """
        self._image_container = self._image_container.mean(axis=1).reshape(-1,
                                                                           1,
                                                                           self.image_dimensions()[1],
                                                                           self.image_dimensions()[0])
        return self

    def convert_to_od(self):
        """
        Converts images in container to optical density maps. Updates image_container automatically.
        """
        absorption_image = self._image_container[:, ::3, :, :]
        probe_image = self._image_container[:, 1::3, :, :]
        background_image = self._image_container[:, 2::3, :, :]
        numerator = probe_image - background_image
        denominator = absorption_image - background_image
        numerator[numerator <= 0] = 1.0
        denominator[denominator <= 0] = 1.0
        self._image_container = np.log(numerator / denominator)
        return self

    def remove_negative_pixels(self):
        self._image_container[self._image_container < 0.0] = 0.0
        return self

    def subtract_constant_offset(self, offset):
        self._image_container = self._image_container - offset
        return self

    def image_dimensions(self):
        """
        Returns dimensions of an image inside _image_container.
        """
        return self._image_container.shape[-1], self._image_container.shape[-2]

    def project_onto_single_axis(self, axis='x'):
        """
        Projects images along x and y axes. Returns both projections. Does not modify _image_container.
        """
        if axis == 'x':
            projection = self._image_container.sum(axis=-2)
        elif axis == 'y':
            projection = self._image_container.sum(axis=-1)
        return projection

    def make_x_and_y(self):
        """
        Make coordinate arrays to match image shape, pixel size, binning and magnification.
        """
        s = self._image_container.shape
        x = np.arange(0, s[-1] * ImageProcessor.pixel_size * self._pixel_binning / self._magnification,
                      ImageProcessor.pixel_size * self._pixel_binning / self._magnification)
        y = np.arange(0, s[-2] * ImageProcessor.pixel_size * self._pixel_binning / self._magnification,
                      ImageProcessor.pixel_size * self._pixel_binning / self._magnification)
        return x, y

    def make_xy_mesh(self):
        """
        Returns a mesh of x and y coordinates. Automatically adjusts for specified imaging parameters.
        Used for setting up a fit of the image to a 2D Gaussian.
        """
        x = self.make_x_and_y()[0]
        y = self.make_x_and_y()[1]
        return np.meshgrid(x, y)

    def integrate_over_image(self):
        """
        Integrates over images and returns total raw signal in each image.
        """
        return self._image_container.sum(axis=(-1, -2))

    def total_signal(self):
        """
        Returns the mean of the total integrated signal and the standard error in each image.
        """
        n = self.integrate_over_image()
        n_mean = n.mean(axis=1)
        n_err = n.std(axis=1) / np.sqrt(self._image_container.shape[1])
        return n_mean, n_err

    def cloud_shape(self, method='1d_gauss', size_or_pos='size', return_all = False):
        """

        Parameters
        ----------
        method: '1d_gauss' or '2d_gauss'. Method used to find cloud shape. If '1d_gauss', the images are first
        integrated along x and y and fitted to a 1D Gaussian. The fitting routine has a helper function which provides
        with initial guesses for the fit parameters. If '2d_gauss', the images are fitted to a 2D Gaussian. The fitting
        routine must receive initial guesses of the fit parameters. Here, these location and size of the cloud are
        passed to the routine automatically. These are obtained from a 1D Gaussian fit.
        size_or_pos: 'size' or 'pos'. If 'size', returns two lists: first contains sizes along x, the second contains
        sizes along y. If 'pos', returns positions in the same dimesnional order.
        return_all: if True, return all of the fit parameters. For method='1d_gauss' the parameter order is
        [amplitude, offset, position, sigma]. For method='2d_gauss' the parameter order is
        [amplitude, xo, yo, sigma_x, sigma_y, theta, offset].
        """
        x, y = self.make_x_and_y()[0], self.make_x_and_y()[1]
        x_projections = self.project_onto_single_axis(axis='x')
        y_projections = self.project_onto_single_axis(axis='y')
        x_params_1d_gauss = fit_gauss_1d(x, x_projections)
        y_params_1d_gauss = fit_gauss_1d(y, y_projections)
        if method == '1d_gauss':
            if return_all:
                return x_params_1d_gauss, y_params_1d_gauss
            elif size_or_pos == 'size':
                return x_params_1d_gauss[:, :, 3], y_params_1d_gauss[:, :, 3]
            else:
                return x_params_1d_gauss[:, :, 2], y_params_1d_gauss[:, :, 2]
        elif method == '2d_gauss':
            xy = self.make_xy_mesh()
            data = self._image_container
            guess_x = x_params_1d_gauss[:, :, 2:]
            guess_y = y_params_1d_gauss[:, :, 2:]
            parameters = fit_gauss_2d(xy, data, guess_x, guess_y)
            if return_all:
                return parameters
            elif size_or_pos == 'size':
                return parameters[:, :, 3], parameters[:, :, 4]
            else:
                return parameters[:, :, 1], parameters[:, :, 2]

    def reshape_to_mot_lifetime(self, time_steps, first_point, last_point, normalize=True):
        """
        Method to transform image_container to the shape of MOT lifetime measurement experiments.
        Extremely custom but was useful!
        """
        image_shape = self._image_container.shape
        n_experiments = image_shape[0]
        n_images = image_shape[1]
        dim_1 = image_shape[2]
        dim_2 = image_shape[3]
        reps = n_images // time_steps
        reshaped_im = self._image_container.reshape(-1, reps, time_steps, dim_1, dim_2)
        sum_bg = self._background_container.reshape(-1, reps, time_steps, dim_1, dim_2).sum(axis=(-1, -2)).mean(axis=1)
        sum_bg_reshaped = sum_bg.reshape(-1, 1, time_steps)
        reshaped_im = reshaped_im[:, :, first_point:last_point, :, :]
        sum_bg_reshaped = sum_bg_reshaped[:, :, first_point:last_point]
        sum_reshaped = reshaped_im.sum(axis=(-1, -2)) - sum_bg_reshaped
        if normalize:
            norm = sum_reshaped.max(axis=-1).reshape(n_experiments, reps, -1)
            sum_reshaped = sum_reshaped / norm
        return sum_reshaped

    @property
    def image_container(self):
        if len(self._image_container) == 0:
            raise Exception('Nothing inside!')
        return self._image_container

    @property
    def background_container(self):
        if len(self._background_container) == 0:
            raise Exception('Nothing inside!')
        return self._background_container

    @property
    def get_average(self):
        return self._image_container.mean(axis=1)

    @property
    def total_n_absorption(self):
        if self.image_type != 'Absorption':
            raise Exception('Image is not absorption.')
        pixel_area = (ImageProcessor.pixel_size * self._pixel_binning / self._magnification) ** 2
        sigma = ImageProcessor.sigma_0 / (1 + 4 * (ImageProcessor.detuning / 6.05) ** 2)
        alpha = pixel_area / sigma
        n_mean, n_err = self.total_signal()
        return n_mean * alpha, n_err * alpha

    @property
    def total_n_fluorescence(self):
        if self.image_type != 'Fluorescence':
            raise Exception('Image is not fluorescence.')
        alpha = 1.0
        n_mean, n_err = self.total_signal()
        return n_mean * alpha, n_err * alpha

    @property
    def pixel_binning(self):
        return self._pixel_binning

    @pixel_binning.setter
    def pixel_binning(self, value):
        if type(value) is not int:
            raise Exception('Bin size must be integer')
        self._pixel_binning = value

    @property
    def magnification(self):
        return self._magnification

    @magnification.setter
    def magnification(self, value):
        self._magnification = value
