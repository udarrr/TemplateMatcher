let cv: any;

try {
  cv = require('opencv4nodejs-prebuilt-install');
} catch {}

import { Image, ImageFinderInterface, imageResource, MatchRequest, MatchResult, Region, screen } from '@nut-tree/nut-js';
import { MatchedResults, MatchTemplate, MethodEnum, MethodNameType } from './match-image.function';
import { ScaleImage } from './scale-image.function';
import { ImageProcessor } from './image-processor.class';
import { Mat } from 'opencv4nodejs-prebuilt-install/lib/typings/Mat';

export type CustomOptionsType = { methodType?: MethodNameType; searchMultipleScales?: boolean; scaleSteps?: Array<number>; debug?: boolean; roi?: Region };

export default class TemplateMatchingFinder implements ImageFinderInterface {
  private _config: Partial<MatchRequest<Image | string, CustomOptionsType>>;

  constructor() {
    this._config = { confidence: 0.8, providerData: { searchMultipleScales: true, scaleSteps: [1, 0.9, 0.8, 0.7, 0.6, 0.5], methodType: MethodEnum.TM_CCOEFF_NORMED, debug: false } };
  }

  getConfig() {
    return this._config;
  }

  setConfig(config: CustomOptionsType) {
    this._config = { ...this._config, ...config };
  }

  private async loadNeedle(image: Image | string): Promise<{ data: Mat }> {
    if (typeof image !== 'string') {
      return { data: await ImageProcessor.fromImageWithAlphaChannel(image) };
    } else {
      return { data: await ImageProcessor.fromImageWithAlphaChannel(await imageResource(image)) };
    }
  }

  private async loadHaystack(
    image?: Image | string,
    roi?: Region,
  ): Promise<{
    data: Mat;
    rect: Region | undefined;
    pixelDensity: {
      scaleX: number;
      scaleY: number;
    };
  }> {
    if (typeof image !== 'string' && image) {
      let validRoi = roi ? ImageProcessor.determineMatRectROI(image, this.getIncreasedRectByPixelDensity(roi, image.pixelDensity)) : undefined;

      return { data: await ImageProcessor.fromImageWithAlphaChannel(image, validRoi), rect: validRoi ? ImageProcessor.determineRegionRectROI(validRoi) : undefined, pixelDensity: image.pixelDensity };
    } else {
      if (!image) {
        const imageObj = await screen.grab();
        let validRoi = roi ? ImageProcessor.determineMatRectROI(imageObj, this.getIncreasedRectByPixelDensity(roi, imageObj.pixelDensity)) : undefined;
        const mat = await ImageProcessor.fromImageWithAlphaChannel(imageObj, validRoi);

        return { data: mat, rect: validRoi ? ImageProcessor.determineRegionRectROI(validRoi) : undefined, pixelDensity: imageObj.pixelDensity };
      } else {
        const imageObj = await imageResource(image);
        let validRoi = roi ? ImageProcessor.determineMatRectROI(imageObj, this.getIncreasedRectByPixelDensity(roi, imageObj.pixelDensity)) : undefined;

        return {
          data: await ImageProcessor.fromImageWithAlphaChannel(imageObj, validRoi),
          rect: validRoi ? ImageProcessor.determineRegionRectROI(validRoi) : undefined,
          pixelDensity: imageObj.pixelDensity,
        };
      }
    }
  }

  private throwOnTooLargeNeedle(haystack: Mat, needle: Mat, smallestScaleFactor: number) {
    const scaledRows = smallestScaleFactor * needle.rows;
    const scaledCols = smallestScaleFactor * needle.cols;

    if (scaledRows > haystack.rows || scaledCols > haystack.cols) {
      throw new Error('Search input is too large, try using a smaller template image.');
    }
  }

  private async initData<T extends CustomOptionsType>(matchRequest: MatchRequest<Image | string, T>) {
    const confidence =
      matchRequest.providerData && matchRequest.providerData?.methodType === MethodEnum.TM_SQDIFF && matchRequest.confidence === 0.99
        ? 0.998
        : (matchRequest.providerData && matchRequest.providerData?.methodType === MethodEnum.TM_CCOEFF_NORMED) ||
          (matchRequest.providerData && matchRequest.providerData?.methodType === MethodEnum.TM_CCORR_NORMED && matchRequest.confidence === 0.99)
        ? (this._config.confidence as number)
        : matchRequest.confidence === 0.99 || typeof matchRequest.confidence === 'undefined'
        ? (this._config.confidence as number)
        : matchRequest.confidence;
    const searchMultipleScales = matchRequest.providerData?.searchMultipleScales ? matchRequest.providerData?.searchMultipleScales : this._config.providerData?.searchMultipleScales;
    const scaleSteps = matchRequest.providerData?.scaleSteps || (this._config.providerData?.scaleSteps as Array<number>);
    const methodType = matchRequest.providerData?.methodType || (this._config.providerData?.methodType as MethodNameType);
    const debug = matchRequest.providerData?.debug || (this._config.providerData?.debug as boolean);

    const needle = await this.loadNeedle(matchRequest.needle);
    if (!needle || needle.data.empty) {
      throw new Error(`Failed to load ${typeof matchRequest.needle === 'string' ? matchRequest.needle : matchRequest.needle.id}, got empty image.`);
    }
    const haystack = await this.loadHaystack(matchRequest.haystack, matchRequest.providerData?.roi);
    if (!haystack || haystack.data.empty) {
      throw new Error(
        `Failed to load ${
          matchRequest && matchRequest.haystack && typeof matchRequest.haystack === 'string' && !matchRequest.haystack ? matchRequest.haystack : (matchRequest.haystack as Image).id
        }, got empty image.`,
      );
    }
    if (matchRequest.providerData?.searchMultipleScales) {
      this.throwOnTooLargeNeedle(haystack.data, needle.data, scaleSteps[scaleSteps.length - 1]);
    }

    return {
      haystack: haystack,
      needle: needle,
      confidence: confidence,
      scaleSteps: scaleSteps,
      methodType: methodType,
      debug: debug,
      searchMultipleScales: searchMultipleScales,
      roi: matchRequest.providerData?.roi,
    };
  }

  public async findMatches<CustomOptionsType>(matchRequest: MatchRequest<Image | string, CustomOptionsType>): Promise<MatchResult<Region>[]> {
    let matchResults: Array<MatchResult<Region>> = [];
    let { haystack, needle, confidence, scaleSteps, methodType, debug, searchMultipleScales, roi } = await this.initData(matchRequest as any);

    if (!searchMultipleScales) {
      const overwrittenResults = await MatchTemplate.matchImagesByWriteOverFounded(haystack.data, needle.data, confidence, methodType, debug);
      matchResults.push(...overwrittenResults.results);
    } else {
      const scaledResults = await this.searchMultipleScales(haystack.data, needle.data, confidence, scaleSteps, methodType, debug);
      matchResults.push(...scaledResults);
    }
    return await this.getValidatedMatches(matchResults, haystack.pixelDensity, confidence, roi);
  }

  private getIncreasedRectByPixelDensity(rect: Region, pixelDensity: { scaleX: number; scaleY: number }) {
    rect.left *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;
    rect.width *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;
    rect.top *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;
    rect.height *= pixelDensity.scaleX && pixelDensity.scaleY && pixelDensity.scaleX === pixelDensity.scaleY ? pixelDensity.scaleX : 1;

    return rect;
  }

  private getDecreasedRectByPixelDensity(matchResults: MatchResult<Region>[], pixelDensity: { scaleX: number; scaleY: number }) {
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

  private async getValidatedMatches(
    matchResults: Array<MatchResult<Region>>,
    pixelDensity: {
      scaleX: number;
      scaleY: number;
    },
    confidence: number,
    roi: Region | undefined,
  ) {
    if (!roi) {
      matchResults = this.getDecreasedRectByPixelDensity(matchResults, pixelDensity);
    } else {
      matchResults = matchResults.map((m) => {
        return { confidence: m.confidence, error: m.error, location: new Region(m.location.left + roi.left, m.location.top + roi.top, m.location.width, m.location.height) };
      });
      matchResults = this.getDecreasedRectByPixelDensity(matchResults, pixelDensity);
      ImageProcessor.validateSearchRegion(new Region(Number(roi.left), Number(roi.top), Number(roi.width), Number(roi.height)), new Region(0, 0, await screen.width(), await screen.height()));
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

  public async findMatch<CustomOptionsType>(matchRequest: MatchRequest<Image | string, CustomOptionsType>): Promise<MatchResult<Region>> {
    let { haystack, needle, confidence, scaleSteps, methodType, debug, searchMultipleScales, roi } = await this.initData(matchRequest as any);

    if (!searchMultipleScales) {
      const matches = await MatchTemplate.matchImages(haystack.data, needle.data, methodType, debug);
      const result = await this.getValidatedMatches([matches.data], haystack.pixelDensity, confidence, roi);

      return result[0];
    } else {
      const scaledResults = await this.searchMultipleScales(haystack.data, needle.data, confidence, scaleSteps, methodType, debug, true);

      return (await this.getValidatedMatches(scaledResults.length ? [scaledResults[0]] : scaledResults, haystack.pixelDensity, confidence, roi))[0];
    }
  }

  private async searchMultipleScales(haystack: Mat, needle: Mat, confidence: number, scaleSteps: Array<number>, methodType: MethodNameType, debug: boolean, firstMach: boolean = false) {
    const results: MatchResult<Region>[] = [];

    const needleData = await this.scaleNeedle(haystack, needle, confidence, scaleSteps, methodType, debug, firstMach);
    results.push(...needleData.results);

    if (firstMach && results.length) {
      return results;
    }
    const haystackData = await this.scaleHaystack(needleData.haystack, needle, confidence, scaleSteps, methodType, debug, firstMach);
    results.push(...haystackData.results);

    if (firstMach && results.length) {
      return results;
    }
    return results;
  }

  private async scaleHaystack(haystack: Mat, needle: Mat, confidence: number, scaleSteps: Array<number>, methodType: MethodNameType, debug: boolean, firstMach: boolean = false) {
    const results: MatchResult<Region>[] = [];
    let overWrittenScaledHaystackResult = { results: results, haystack: haystack };
    let overwrittenHaystack = haystack;

    for (const currentScale of scaleSteps) {
      let scaledHaystack = await ScaleImage.scaleImage(overwrittenHaystack, currentScale);

      if (scaledHaystack.cols <= 10 || scaledHaystack.rows <= 10) {
        break;
      }
      if (scaledHaystack.cols * scaledHaystack.rows === 0) {
        break;
      }
      if (scaledHaystack.cols < needle.cols || scaledHaystack.rows < needle.rows) {
        break;
      }
      overWrittenScaledHaystackResult = await MatchTemplate.matchImagesByWriteOverFounded(scaledHaystack, needle, confidence, methodType, debug, firstMach);
      overwrittenHaystack = overWrittenScaledHaystackResult.haystack;
      results.push(...overWrittenScaledHaystackResult.results);
    }
    return { results: results, haystack: overWrittenScaledHaystackResult.haystack };
  }

  private async scaleNeedle(
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
      const scaledNeedle = await ScaleImage.scaleImage(needle, currentScale);

      if (scaledNeedle.cols <= 10 || scaledNeedle.rows <= 10) {
        break;
      }
      if (scaledNeedle.cols * scaledNeedle.rows === 0) {
        break;
      }
      if (haystack.cols < scaledNeedle.cols || haystack.rows < scaledNeedle.rows) {
        break;
      }

      overWrittenScaledNeedleResult = await MatchTemplate.matchImagesByWriteOverFounded(haystack, scaledNeedle, confidence, methodType, debug, firstMatch);
      results.push(...overWrittenScaledNeedleResult.results);
    }
    return { results: results, haystack: overWrittenScaledNeedleResult.haystack };
  }
}
