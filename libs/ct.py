import numpy as np
from libs.utils import resize_to_square, image_center, pad_image
import cv2
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

class CT:
    def __init__(self, image, angle_range, n_scans, n_detectors, filter=False, visualize=False, mask=False, test=False, mid=False):
        self.image = pad_image(resize_to_square(image), 50)
        self.angle_range = angle_range
        self.n_scans = n_scans
        self.n_detectors = n_detectors
        self.alphas = np.linspace(0, 180, self.n_scans)
        self.filter = filter
        self.visualize = visualize
        self.mask = mask
        self.test = test

        self.sinogram = None
        self.filtered_sinogram = None
        self.reconstructed_image = None

        self.tested_rsme = []

        self.mid = mid
        self.mid_photos = []
        self.mid_rsme = []
        self.mid_values = []

    def _generate_points(self, image, alpha):
        r, center = image.shape[0]//2, image_center(image)

        angles_points = np.linspace(-self.angle_range/2, self.angle_range/2, self.n_detectors) + alpha
        xs = r * np.cos(np.radians(angles_points)) + center[0]
        ys = r * np.sin(np.radians(angles_points)) + center[1]
        return np.array(list(zip(xs, ys))).astype(int)

    def generate_detectors(self, image, alpha):
        return self._generate_points(image, alpha)

    def generate_emitters(self, image, alpha):
        return np.flip(self.generate_detectors(image, alpha + 180), 0)

    def _bresenham_line(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            # Add current point
            points.append((x1, y1))

            # Check if reached the end
            if x1 == x2 and y1 == y2:
                break

            # Calculate next point
            err2 = 2 * err
            if err2 > -dy:
                err -= dy
                x1 += sx
            if err2 < dx:
                err += dx
                y1 += sy

        return points

    def _generate_image_lines(self, image, alpha):
        emitters = self.generate_emitters(image, alpha)
        detectors = self.generate_detectors(image, alpha)

        lines = [np.array(self._bresenham_line(emitter, detector)) for emitter, detector in zip(emitters, detectors)]

        for line in lines:
            line[line < 0] = 0
            line[line[:, 0] >= image.shape[0], 0] = image.shape[0] - 1
            line[line[:, 1] >= image.shape[1], 1] = image.shape[1] - 1

        lines = map(lambda line: np.array(line).T, lines)
        lines = list(lines)

        return lines

    def _radon_transform(self):
        results = np.zeros((self.n_scans, self.n_detectors))
        for i, alpha in enumerate(self.alphas):
            lines = self._generate_image_lines(self.image, alpha)
            results[i] = np.array([np.average(self.image[tuple(line)]) for line in lines])

        return results.T

    def _inverse_radon_transform(self):
        results = np.zeros((self.image.shape[0], self.image.shape[1]))
        if self.filter:
            sinogram = self._filter()
            self.filtered_sinogram = sinogram
        else:
            sinogram = self.sinogram

        for i, alpha in enumerate(self.alphas):
            lines = self._generate_image_lines(results, alpha)
            for j, line in enumerate(lines):
                results[tuple(line)] += sinogram[j, i]

            if self.test:
                self.tested_rsme.append(np.sqrt(np.mean((self.image - results)**2)))

            if self.mid:
                if i%45 == 0:
                    self.mid_photos.append(results.copy())
                    self.mid_rsme.append(np.sqrt(np.mean((self.image - results)**2)))
                    self.mid_values.append(i)

        return results

    def _visualize_images(self, images, labels):
        _, axes = plt.subplots(1, len(images), figsize=(10, 10))
        for ax, image, label in zip(axes, images, labels):
            ax.imshow(image, cmap='gray')
            ax.axis('off')
            ax.set_title(label)
        plt.show()

    def _filter(self):
        filter = 2 * np.abs(fftfreq(self.n_detectors).reshape(-1, 1))
        result = ifft(fft(self.sinogram, axis=0) * filter, axis=0)
        return np.real(result)

    def rmse(self):
        return np.sqrt(np.mean((self.image - self.reconstructed_image)**2))

    def apply_mask(self):
        mask = np.zeros_like(self.reconstructed_image)
        mask = cv2.circle(mask, (self.reconstructed_image.shape[0]//2, self.reconstructed_image.shape[1]//2), self.reconstructed_image.shape[0]//2, (255, 255, 255), -1)
        self.reconstructed_image = cv2.bitwise_and(self.reconstructed_image, mask)

        # self.reconstructed_image = cv2.equalizeHist(self.reconstructed_image.astype(np.uint8))
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(256, 256))
        # self.reconstructed_image = clahe.apply(self.reconstructed_image.astype(np.uint8))

        if self.mid:
            self.mid_photos = [cv2.bitwise_and(photo, mask) for photo in self.mid_photos]

        return self.reconstructed_image, self.rmse()

    def return_mid(self):
        return self.mid_photos, self.mid_rsme, self.mid_values

    def scan(self):
        self.sinogram = self._radon_transform()
        self.reconstructed_image = self._inverse_radon_transform()

        if self.mask:
            self.apply_mask()

        if self.visualize:
            if self.filter:
                self._visualize_images([self.image, resize_to_square(self.sinogram), resize_to_square(self.filtered_sinogram), self.reconstructed_image], ['Original Image', 'Original Sinogram', 'Filtered Sinogram', 'Reconstructed Image'])
            else:
                self._visualize_images([self.image, resize_to_square(self.sinogram), self.reconstructed_image], ['Original Image', 'Sinogram', 'Reconstructed Image'])
        return self.reconstructed_image, self.rmse()

def generate_points(image, alpha, angle_range, n_points):
    r, center = image.shape[0]//2, image_center(image)

    angles_points = np.linspace(-angle_range/2, angle_range/2, n_points) + alpha
    xs = r * np.cos(np.radians(angles_points)) + center[0]
    ys = r * np.sin(np.radians(angles_points)) + center[1]
    return np.floor(np.array(list(zip(xs, ys)))).astype(int)

def generate_detectors(image, alpha, angle_range, n_detectors):
    return generate_points(image, alpha, angle_range, n_detectors)

def generate_emitters(image, alpha, angle_range, emitters):
    return np.flip(generate_points(image, alpha + 180, angle_range, emitters), 0)

def draw_emitters_detectors(image, alpha, angle_range, detectors):

    emitters = generate_emitters(image, alpha, angle_range, detectors)
    detectors = generate_detectors(image, alpha, angle_range, detectors)

    plt.figure(figsize=(5,5))

    # Tworzenie obrazu
    plt.imshow(image, cmap='gray')

    # Rysowanie pozycji emiterów i detektorów
    plt.plot(emitters[:,0], emitters[:,1], 'bo', label='Emitters')
    plt.plot(detectors[:,0], detectors[:,1], 'ro', label='Detectors')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Emitter and Detector Positions on Image')
    plt.legend()
    plt.axis('off')

    plt.show()

def visualize_experiments(results, title):
    plt.figure(figsize=(20, 10))
    plt.suptitle(title)
    for i, result in enumerate(results):
        plt.subplot(2, 4, i+1)
        plt.imshow(result['image'], cmap='gray')
        plt.title(f"{result['testedVar']}: {result['testedValue']}, RMSE: {result['rmse']:.2f}")
        plt.axis('off')
    plt.show()

    #plot rmse of values
    plt.figure(figsize=(5, 3))
    plt.suptitle(f"RMSE of {title}")
    plt.plot([result['testedValue'] for result in results], [result['rmse'] for result in results])
    plt.xlabel(title)
    plt.ylabel("RMSE")
    plt.show()
