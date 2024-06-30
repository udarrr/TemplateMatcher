import * as path from 'path';
import { MatchRequest, Region } from '@nut-tree-fork/nut-js';
import ImageReader from '../lib/readers/imageReader.class';
import TemplateMatchingFinder from '../lib/templateMatchingFinder';

jest.mock('jimp', () => {});

describe('Template-matching finder', () => {
  it('findMatch should return a match when present in image', async () => {
    // GIVEN
    const imageLoader = new ImageReader();
    const SUT = new TemplateMatchingFinder();
    const haystackPath = path.resolve(__dirname, './__mocks__/mouse.png');
    const needlePath = path.resolve(__dirname, './__mocks__/needle.png');
    const haystack = await imageLoader.load(haystackPath);
    const needle = await imageLoader.load(needlePath);
    const minConfidence = 0.99;
    const matchRequest = new MatchRequest(haystack, needle, minConfidence);
    const expectedResult = new Region(0, 0, needle.width, needle.height);

    // WHEN
    const result = await SUT.findMatch(matchRequest);

    // THEN
    expect(result.confidence).toBeGreaterThanOrEqual(minConfidence);
    expect(result.location).toEqual(expectedResult);
  });

  it('findMatch should return confidence and location of best match if no match with sufficient confidence is found', async () => {
    // GIVEN
    const imageLoader = new ImageReader();
    const SUT = new TemplateMatchingFinder();
    const haystackPath = path.resolve(__dirname, './__mocks__/downloads.png');
    const needlePath = path.resolve(__dirname, './__mocks__/coverage.png');
    const haystack = await imageLoader.load(haystackPath);
    const needle = await imageLoader.load(needlePath);
    const minConfidence = 0.99;
    const matchRequest = new MatchRequest(haystack, needle, minConfidence);
    const expectedRejection = new RegExp(`Unable to locate on screen with template, no match!`);

    // WHEN

    // THEN
    await expect(SUT.findMatch(matchRequest)).rejects.toThrowError(expectedRejection);
  });

  it('findMatch should reject, if needle was way larger than haystack', async () => {
    // GIVEN
    const imageLoader = new ImageReader();
    const SUT = new TemplateMatchingFinder();
    const haystackPath = path.resolve(__dirname, './__mocks__/mouse.png');
    const needlePath = path.resolve(__dirname, './__mocks__/fat-needle.png');
    const haystack = await imageLoader.load(haystackPath);
    const needle = await imageLoader.load(needlePath);
    const minConfidence = 0.99;
    const matchRequest = new MatchRequest(haystack, needle, minConfidence);
    // const expectedRejection = new Error("The provided image sample is larger than the provided search region")

    // WHEN
    const findMatchPromise = SUT.findMatch(matchRequest);

    // THEN
    // await expect(findMatchPromise).rejects.toEqual(expectedRejection)
    await expect(findMatchPromise).rejects.toThrowError('Search input is too large, try using a smaller template image.');
  });
});
