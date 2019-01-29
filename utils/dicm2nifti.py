from nipype.interfaces.dcm2nii import Dcm2niix
import os


if __name__ == '__main__':

    #example to convert dicom to nifti provide the data contain the dicom folder as source_dir see example below

    converter = Dcm2niix();
    converter.inputs.source_dir = '/home/jehill/Documents/ML_PROJECTS/VNETS/DATA'
    converter.cmdline
    converter.run()