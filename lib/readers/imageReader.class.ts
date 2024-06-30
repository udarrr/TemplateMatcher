let cv: any;

try {
  cv = require('opencv4nodejs-prebuilt-install');
} catch {}

import { Image, ImageReader } from '@nut-tree-fork/nut-js';

export default class implements ImageReader {
  public async load(path: string): Promise<Image> {
    return new Promise<Image>(async (resolve, reject) => {
      try {
        const image = await cv.imreadAsync(path, cv.IMREAD_UNCHANGED);
        resolve(new Image(image.cols, image.rows, image.getData(), image.channels, path, 4, image.cols * 4));
      } catch (e) {
        reject(`Failed to load image from '${path}'`);
      }
    });
  }
}
