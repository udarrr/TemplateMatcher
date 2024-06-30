let cv: any;

try {
  cv = require('opencv4nodejs-prebuilt-install');
} catch {}

import { MatchResult, Region } from '@nut-tree-fork/nut-js';
import { MethodNameType, MatchedResults } from '../types';
import { OverWritingMatcherHandler } from './overWriting';
import { Mat } from 'opencv4nodejs-prebuilt-install';

export class ScaleImageHandler {
  static lowerBound(value: number, boundary: number, minValue: number): number {
    return value <= boundary ? minValue : value;
  }

  static upperBound(value: number, boundary: number, maxValue: number): number {
    return value >= boundary ? maxValue : value;
  }

  static async scaleImage(image: Mat, scaleFactor: number): Promise<Mat> {
    const boundScaleFactor = ScaleImageHandler.lowerBound(scaleFactor, 0.0, 1.0);
    const scaledRows = Math.floor(image.rows * boundScaleFactor);
    const scaledCols = Math.floor(image.cols * boundScaleFactor);
    return image.resizeAsync(scaledRows, scaledCols, 0, 0, cv.INTER_AREA);
  }

  static scaleLocation(result: Region, scaleFactor: number): Region {
    const boundScaleFactor = ScaleImageHandler.lowerBound(scaleFactor, 0.0, 1.0);
    return new Region(result.left / boundScaleFactor, result.top / boundScaleFactor, result.width, result.height);
  }

  static async scaleHaystack(haystack: Mat, needle: Mat, confidence: number, scaleSteps: Array<number>, methodType: MethodNameType, debug: boolean, firstMach: boolean = false) {
    const results: MatchResult<Region>[] = [];
    let overWrittenScaledHaystackResult = { results: results, haystack: haystack };
    let overwrittenHaystack = haystack;

    for (const currentScale of scaleSteps) {
      let scaledHaystack = await ScaleImageHandler.scaleImage(overwrittenHaystack, currentScale);

      if (scaledHaystack.cols <= 10 || scaledHaystack.rows <= 10) {
        break;
      }
      if (scaledHaystack.cols * scaledHaystack.rows === 0) {
        break;
      }
      if (scaledHaystack.cols < needle.cols || scaledHaystack.rows < needle.rows) {
        break;
      }
      overWrittenScaledHaystackResult = await OverWritingMatcherHandler.matchImagesByWriteOverFounded(scaledHaystack, needle, confidence, methodType, debug, firstMach);
      overwrittenHaystack = overWrittenScaledHaystackResult.haystack;
      results.push(...overWrittenScaledHaystackResult.results);
    }
    return { results: results, haystack: overWrittenScaledHaystackResult.haystack };
  }

  static async scaleNeedle(
    haystack: Mat,
    needle: Mat,
    confidence: number = 0.8,
    scaleSteps: Array<number>,
    methodType: MethodNameType,
    debug: boolean,
    firstMatch: boolean = false,
  ): Promise<MatchedResults> {
    const results: MatchResult<Region>[] = [];
    let overWrittenScaledNeedleResult = { results: results, haystack: haystack };

    for (const currentScale of scaleSteps) {
      const scaledNeedle = await ScaleImageHandler.scaleImage(needle, currentScale);

      if (scaledNeedle.cols <= 10 || scaledNeedle.rows <= 10) {
        break;
      }
      if (scaledNeedle.cols * scaledNeedle.rows === 0) {
        break;
      }
      if (haystack.cols < scaledNeedle.cols || haystack.rows < scaledNeedle.rows) {
        break;
      }

      overWrittenScaledNeedleResult = await OverWritingMatcherHandler.matchImagesByWriteOverFounded(haystack, scaledNeedle, confidence, methodType, debug, firstMatch);
      results.push(...overWrittenScaledNeedleResult.results);
    }
    return { results: results, haystack: overWrittenScaledNeedleResult.haystack };
  }

  static async searchMultipleScales(haystack: Mat, needle: Mat, confidence: number, scaleSteps: Array<number>, methodType: MethodNameType, debug: boolean, firstMach: boolean = false) {
    const results: MatchResult<Region>[] = [];

    const needleData = await ScaleImageHandler.scaleNeedle(haystack, needle, confidence, scaleSteps, methodType, debug, firstMach);
    results.push(...needleData.results);

    if (firstMach && results.length) {
      return results;
    }
    const haystackData = await ScaleImageHandler.scaleHaystack(needleData.haystack, needle, confidence, scaleSteps, methodType, debug, firstMach);
    results.push(...haystackData.results);

    if (firstMach && results.length) {
      return results;
    }
    return results;
  }
}
