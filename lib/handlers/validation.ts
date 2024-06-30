let cv: any;

try {
  cv = require('opencv4nodejs-prebuilt-install');
} catch {}

import { Image, MatchResult, Region, screen } from '@nut-tree-fork/nut-js';
import { Mat, Rect } from 'opencv4nodejs-prebuilt-install';

export class ValidationHandler {
  static determineMatRectROI(img: Image, roi: Region): Rect {
    return new cv.Rect(Math.min(Math.max(roi.left, 0), img.width), Math.min(Math.max(roi.top, 0), img.height), Math.min(roi.width, img.width - roi.left), Math.min(roi.height, img.height - roi.top));
  }

  static determineRegionRectROI(roi: Rect): Region {
    return new Region(roi.x, roi.y, roi.width, roi.height);
  }

  static validateSearchRegion(search: Region, screen: Region) {
    if (search.left < 0 || search.top < 0 || search.width < 0 || search.height < 0) {
      throw new Error(`Negative values in search region ${search}`);
    }
    if (isNaN(search.left) || isNaN(search.top) || isNaN(search.width) || isNaN(search.height)) {
      throw new Error(`NaN values in search region ${search}`);
    }
    if (search.width < 2 || search.height < 2) {
      throw new Error(`Search region ${search} is not large enough. Must be at least two pixels in both width and height.`);
    }
    if (search.left + search.width > screen.width || search.top + search.height > screen.height) {
      throw new Error(`Search region ${search} extends beyond screen boundaries (${screen.width}x${screen.height})`);
    }
  }

  static getIncreasedRectByPixelDensity(rect: Region, pixelDensity: { scaleX: number; scaleY: number }) {
    rect.left *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;
    rect.width *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;
    rect.top *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;
    rect.height *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;

    return rect;
  }

  static getDecreasedRectByPixelDensity(matchResults: MatchResult<Region>[], pixelDensity: { scaleX: number; scaleY: number }) {
    return matchResults.map((results) => {
      return {
        confidence: results.confidence,
        error: results.error,
        location: new Region(
          (results.location.left /= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1),
          (results.location.top /= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1),
          (results.location.width /= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1),
          (results.location.height /= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1),
        ),
      };
    });
  }

  static throwOnTooLargeNeedle(haystack: Mat, needle: Mat, smallestScaleFactor: number) {
    const scaledRows = smallestScaleFactor * needle.rows;
    const scaledCols = smallestScaleFactor * needle.cols;

    if (scaledRows > haystack.rows || scaledCols > haystack.cols) {
      throw new Error('Search input is too large, try using a smaller template image.');
    }
  }

  static async getValidatedMatches(
    matchResults: Array<MatchResult<Region>>,
    pixelDensity: {
      scaleX: number;
      scaleY: number;
    },
    confidence: number,
    roi: Region | undefined,
  ) {
    if (!roi) {
      matchResults = ValidationHandler.getDecreasedRectByPixelDensity(matchResults, pixelDensity);
    } else {
      matchResults = matchResults.map((m) => {
        return { confidence: m.confidence, error: m.error, location: new Region(m.location.left + roi.left, m.location.top + roi.top, m.location.width, m.location.height) };
      });
      matchResults = ValidationHandler.getDecreasedRectByPixelDensity(matchResults, pixelDensity);
      ValidationHandler.validateSearchRegion(new Region(Number(roi.left), Number(roi.top), Number(roi.width), Number(roi.height)), new Region(0, 0, await screen.width(), await screen.height()));
    }
    matchResults.sort((first, second) => second.confidence - first.confidence);

    const potentialMatches = matchResults.filter((match) => match.confidence >= confidence);

    if (potentialMatches.length === 0) {
      matchResults.sort((a, b) => a.confidence - b.confidence);
      const bestMatch = matchResults.pop();

      if (bestMatch) {
        throw new Error(`No match with required confidence ${confidence}. Best match: ${bestMatch.confidence}`);
      } else {
        throw new Error(`Unable to locate on screen with template, no match!`);
      }
    }
    return potentialMatches;
  }
}
