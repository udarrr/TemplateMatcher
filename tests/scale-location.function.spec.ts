import { Region } from '@nut-tree-fork/nut-js';
import { ScaleImageHandler } from '../lib/handlers/scaleImage';

jest.mock('jimp', () => {});

describe('scaleLocation', () => {
  it('should scale location of a Region for valid scale factors', () => {
    // GIVEN
    const scaleFactor = 0.5;
    const inputRegion = new Region(100, 100, 10, 10);
    const expectedRegion = new Region(200, 200, 10, 10);

    // WHEN
    const result = ScaleImageHandler.scaleLocation(inputRegion, scaleFactor);

    // THEN
    expect(result).toEqual(expectedRegion);
  });

  it('should not scale location of a Region for invalid scale factors', () => {
    // GIVEN
    const scaleFactor = 0.0;
    const inputRegion = new Region(100, 100, 10, 10);
    const expectedRegion = new Region(100, 100, 10, 10);

    // WHEN
    const result = ScaleImageHandler.scaleLocation(inputRegion, scaleFactor);

    // THEN
    expect(result).toEqual(expectedRegion);
  });
});
