import pydicom
import matplotlib.pyplot as plt
import pydicom._storage_sopclass_uids
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

def read_dicom(file_path):
    dataset = pydicom.dcmread(file_path)

    plt.imshow(dataset.pixel_array, cmap='gray')
    plt.axis('off')
    plt.show()

    return dataset

def write_dicom(img, metadata):
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    cimg = img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))

    fds = pydicom.FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    fds.is_little_endian = True
    fds.is_implicit_VR = False
    fds.SeriesInstanceUID = pydicom.uid.generate_uid()
    fds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    fds.Modality = "CT"
    fds.SamplesPerPixel = 1
    fds.PhotometricInterpretation = "MONOCHROME2"
    fds.PixelData = cimg
    fds.BitsStored = 8
    fds.BitsAllocated = 8
    fds.HighBit = 7
    fds.ImagesInAcquisition = 1
    fds.Rows, fds.Columns = cimg.shape
    fds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    fds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    fds.StudyInstanceUID = pydicom.uid.generate_uid()
    fds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    fds.PixelRepresentation = 0
    fds.InstanceNumber = 1

    for key, value in metadata.items():
        setattr(fds, key, value)

    pydicom.dataset.validate_file_meta(fds.file_meta, enforce_standard=True)
    fds.save_as(metadata['PatientName'] + ".dcm", write_like_original=False)

    return fds
