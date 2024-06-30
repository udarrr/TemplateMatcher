let cv: any;

try {
  cv = require('opencv4nodejs-prebuilt-install');
} catch {}

import { ColorMode, Image } from '@nut-tree-fork/nut-js';
import { Mat } from 'opencv4nodejs-prebuilt-install/lib/typings/Mat';
import { Rect } from 'opencv4nodejs-prebuilt-install/lib/typings/Rect';

export class ImageProcessor {
  static fromImageWithAlphaChannel = async (img: Image, roi?: Rect): Promise<Mat> => {
    let mat: Mat;
    if (img.colorMode === ColorMode.RGB) {
      mat = await new cv.Mat(img.data, img.height, img.width, cv.CV_8UC4).cvtColorAsync(cv.COLOR_RGBA2GRAY);
    } else {
      mat = await new cv.Mat(img.data, img.height, img.width, cv.CV_8UC4).cvtColorAsync(cv.COLOR_BGRA2GRAY);
    }
    if (roi) {
      return mat.getRegion(roi);
    } else {
      return mat;
    }
  };

  static fromImageWithoutAlphaChannel = async (img: Image, roi?: Rect): Promise<Mat> => {
    let mat: Mat;
    if (img.colorMode === ColorMode.RGB) {
      mat = await new cv.Mat(img.data, img.height, img.width, cv.CV_8UC3).cvtColorAsync(cv.COLOR_RGB2GRAY);
    } else {
      mat = await new cv.Mat(img.data, img.height, img.width, cv.CV_8UC3).cvtColorAsync(cv.COLOR_BGR2GRAY);
    }
    if (roi) {
      return mat.getRegion(roi);
    } else {
      return mat;
    }
  };
}
