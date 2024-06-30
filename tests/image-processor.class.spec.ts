import { resolve } from 'path';
import { Region } from '@nut-tree-fork/nut-js';
import { ImageProcessor } from '../lib/readers/imageProcessor.class';
import ImageReader from '../lib/readers/imageReader.class';
import { ValidationHandler } from '../lib/handlers/validation';

jest.mock('jimp', () => {});

describe('ImageProcessor', () => {
  it('should allow to create a cv.Mat from an Image with alpha channel, alpha channel is dropped', async () => {
    // GIVEN
    const imageReader = new ImageReader();
    const imagePath = resolve(__dirname, './__mocks__/alpha_channel.png');
    const image = await imageReader.load(imagePath);

    // WHEN
    const mat = await ImageProcessor.fromImageWithAlphaChannel(image);

    // THEN
    expect(image.hasAlphaChannel).toBeTruthy();
    expect(mat.channels).toEqual(1);
    expect(mat.rows).toEqual(image.height);
    expect(mat.cols).toEqual(image.width);
    expect(mat.empty).toBeFalsy();
  });

  it('should allow to create a cv.Mat from an Image without alpha channel', async () => {
    // GIVEN
    const imageReader = new ImageReader();
    const imagePath = resolve(__dirname, './__mocks__/fat-needle.png');
    const image = await imageReader.load(imagePath);

    // WHEN
    const mat = await ImageProcessor.fromImageWithoutAlphaChannel(image);

    // THEN
    expect(image.hasAlphaChannel).toBeFalsy();
    expect(mat.channels).toEqual(1);
    expect(mat.rows).toEqual(image.height);
    expect(mat.cols).toEqual(image.width);
    expect(mat.empty).toBeFalsy();
  });
});

describe('ImageProcessor with ROI', () => {
  it('negative left or top values are updated to 0', async () => {
    // GIVEN
    const imageReader = new ImageReader();
    const imagePath = resolve(__dirname, './__mocks__/fat-needle.png');
    const image = await imageReader.load(imagePath);

    // WHEN
    const mat = await ImageProcessor.fromImageWithoutAlphaChannel(image, ValidationHandler.determineMatRectROI(image, new Region(-100, -100, 10, 10)));

    // THEN
    expect(image.hasAlphaChannel).toBeFalsy();
    expect(mat.channels).toEqual(1);
    expect(mat.rows).toEqual(10);
    expect(mat.cols).toEqual(10);
    expect(mat.empty).toBeFalsy();
  });

  it('values bigger than the input are updated to width and height', async () => {
    // GIVEN
    const imageReader = new ImageReader();
    const imagePath = resolve(__dirname, './__mocks__/fat-needle.png');
    const image = await imageReader.load(imagePath);

    // WHEN
    const mat = await ImageProcessor.fromImageWithoutAlphaChannel(image, ValidationHandler.determineMatRectROI(image, new Region(10, 10, image.width * 2, image.height * 2)));

    // THEN
    expect(image.hasAlphaChannel).toBeFalsy();
    expect(mat.channels).toEqual(1);
    expect(mat.rows).toEqual(image.height - 10);
    expect(mat.cols).toEqual(image.width - 10);
    expect(mat.empty).toBeFalsy();
  });
});
