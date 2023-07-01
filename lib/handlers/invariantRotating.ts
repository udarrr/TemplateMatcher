// Source C++ library //https://github.com/DennisLiu1993/Fastest_Image_Pattern_Matching under BSD 2-Clause License Copyright (c) 2022, DennisLiu1993

import cv from 'opencv4nodejs-prebuilt-install';
import { MatchParameter, SingleTargetMatch, Vector } from '../types';
import { Mat, Size, BORDER_CONSTANT, CV_32F, CV_64F, FILLED, INTER_LINEAR, Point2, Rect, RotatedRect, Vec3 } from 'opencv4nodejs-prebuilt-install';

export class InvariantRotatingHandler {
  private static MATCH_CANDIDATE_NUM = 5;
  private static MAX_VISIBLE = 0.0000001;
  private static INTERSECT_NONE = 0;
  private static INTERSECT_FULL = 0;
  private static D2R = Math.PI / 180;
  private static R2D = 180.0 / Math.PI;

  private static GetTopLayer(matTempl: Mat, iMinDstLength: number) {
    let iTopLayer = 0;
    const iMinReduceArea = iMinDstLength * iMinDstLength;
    let iArea = matTempl.cols * matTempl.rows;

    while (iArea > iMinReduceArea) {
      iArea /= 4;
      iTopLayer++;
    }
    return iTopLayer;
  }

  private static async GetRotatedROI(matSrc: Mat, size: Size, ptLT: Point2, dAngle: number) {
    const dAngle_radian = dAngle * InvariantRotatingHandler.D2R;
    const ptC = new Point2((matSrc.cols - 1) / 2.0, (matSrc.rows - 1) / 2.0);
    const ptLT_rotate = InvariantRotatingHandler.ptRotatePt2f(ptLT, ptC, dAngle_radian).sub(new Point2(3, 3)) as Point2;
    const sizePadding: Size = { width: size.width + 6, height: size.height + 6 };

    const rMat = cv.getRotationMatrix2D(ptC, dAngle, 1);
    rMat.set(0, 2, rMat.at(0, 2) - ptLT_rotate.x);
    rMat.set(1, 2, rMat.at(1, 2) - ptLT_rotate.y);

    return await matSrc.warpAffineAsync(rMat, new Size(sizePadding.width, sizePadding.height));
  }

  static async Match(
    matSrc: Mat,
    matDst: Mat,
    iMinDstLength: number,
    dScore: number = 0.8,
    rotationRange: number = 180,
    dMaxOverlap: number = 0,
    subPixEstimation: boolean = false,
    debug: boolean = false,
  ) {
    if ((matDst.cols < matSrc.cols && matDst.rows > matSrc.rows) || (matDst.cols > matSrc.cols && matDst.rows < matSrc.rows)) {
      throw new Error(
        `(matDst.cols < matSrc.cols && matDst.rows > matSrc.rows) || (matDst.cols > matSrc.cols && matDst.rows < matSrc.rows) is ${
          (matDst.cols < matSrc.cols && matDst.rows > matSrc.rows) || (matDst.cols > matSrc.cols && matDst.rows < matSrc.rows)
        }`,
      );
    }

    if (matDst.cols * matDst.rows > matSrc.cols * matDst.rows) {
      throw new Error(`matDst.cols * matDst.rows > matSrc.cols * matDst.rows is ${matDst.cols * matDst.rows > matSrc.cols * matDst.rows}`);
    }

    let m_iMaxPos = 70;
    let m_dToleranceAngle = rotationRange;
    let m_dMaxOverlap = dMaxOverlap;
    let m_bStopLayer1 = false;
    let m_bToleranceRange = false;
    let startTimeCheck = 0;
    let bSubPixelEstimation = subPixEstimation;

    if (debug) {
      startTimeCheck = performance.now();
    }
    let iTopLayer = InvariantRotatingHandler.GetTopLayer(matDst, Math.sqrt(iMinDstLength));
    const vecMatSrcPyr = await matSrc.buildPyramidAsync(iTopLayer);
    const pTemplData = await InvariantRotatingHandler.LearnPattern(matDst, iMinDstLength);

    if (!pTemplData.bIsPatternLearned) {
      throw new Error(`bIsPatternLearned is ${pTemplData.bIsPatternLearned}`);
    }

    const dAngleStep = Math.atan(2.0 / Math.max(pTemplData.vecPyramid[iTopLayer].cols, pTemplData.vecPyramid[iTopLayer].rows)) * InvariantRotatingHandler.R2D;

    const vecAngles: Array<number> = [];

    if (m_dToleranceAngle < InvariantRotatingHandler.MAX_VISIBLE) {
      vecAngles.push(0.0);
    } else {
      for (let dAngle = 0; dAngle < m_dToleranceAngle + dAngleStep; dAngle += dAngleStep) {
        vecAngles.push(dAngle);
      }
      for (let dAngle = -dAngleStep; dAngle > -m_dToleranceAngle - dAngleStep; dAngle -= dAngleStep) {
        vecAngles.push(dAngle);
      }
    }
    const iTopSrcW = vecMatSrcPyr[iTopLayer].cols;
    const iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
    const ptCenter = new Point2((iTopSrcW - 1) / 2.0, (iTopSrcH - 1) / 2.0);

    let vecMatchParameter: Array<MatchParameter> = [];
    const vecLayerScore: Array<number> = [...Array(iTopLayer + 1).keys()].fill(dScore);
    const bCalMaxByBlock =
      (vecMatSrcPyr[iTopLayer].sizes[0] * vecMatSrcPyr[iTopLayer].sizes[1]) / (pTemplData.vecPyramid[iTopLayer].sizes[0] * pTemplData.vecPyramid[iTopLayer].sizes[1]) > 500 && m_iMaxPos > 10;

    for (let iLayer = 1; iLayer <= iTopLayer; iLayer++) {
      vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;
    }

    for (let i = 0; i < vecAngles.length; i++) {
      let matR = cv.getRotationMatrix2D(ptCenter, vecAngles[i], 1);
      let sizeBest = InvariantRotatingHandler.GetBestRotationSize(
        { width: vecMatSrcPyr[iTopLayer].cols, height: vecMatSrcPyr[iTopLayer].rows },
        { width: pTemplData.vecPyramid[iTopLayer].cols, height: pTemplData.vecPyramid[iTopLayer].rows },
        vecAngles[i],
      );

      let fTranslationX = (sizeBest.width - 1) / 2.0 - ptCenter.x;
      let fTranslationY = (sizeBest.height - 1) / 2.0 - ptCenter.y;

      matR.set(0, 2, matR.at(0, 2) + fTranslationX);
      matR.set(1, 2, matR.at(1, 2) + fTranslationY);

      const matRotatedSrc = await vecMatSrcPyr[iTopLayer].warpAffineAsync(
        matR,
        new Size(sizeBest.width, sizeBest.height),
        INTER_LINEAR,
        BORDER_CONSTANT,
        new Vec3(pTemplData.iBorderColor, pTemplData.iBorderColor, pTemplData.iBorderColor),
      );
      const matResult = await InvariantRotatingHandler.MatchTemplate(matRotatedSrc, pTemplData, iTopLayer);

      if (bCalMaxByBlock) {
        const blockMax = new SBlockMax(matResult, new Size(pTemplData.vecPyramid[iTopLayer].cols, pTemplData.vecPyramid[iTopLayer].rows));
        const minMaxLoc = blockMax.GetMaxValueLoc();

        let nextMaxLoc: {
          srcMat: Mat;
          minMaxLoc: {
            dMaxVal: number;
            ptMaxLoc: Point2;
          };
        } = { srcMat: matResult, minMaxLoc: minMaxLoc };

        if (minMaxLoc.dMaxVal < vecLayerScore[iTopLayer]) {
          continue;
        }
        vecMatchParameter.push({
          pt: new Point2(nextMaxLoc.minMaxLoc.ptMaxLoc.x - fTranslationX, nextMaxLoc.minMaxLoc.ptMaxLoc.y - fTranslationY),
          size: new Size(pTemplData.vecPyramid[iTopLayer].cols, pTemplData.vecPyramid[iTopLayer].rows),
          dMatchScore: nextMaxLoc.minMaxLoc.dMaxVal,
          dMatchAngle: vecAngles[i],
        });

        for (let j = 0; j < m_iMaxPos + InvariantRotatingHandler.MATCH_CANDIDATE_NUM - 1; j++) {
          nextMaxLoc = await InvariantRotatingHandler.GetNextMaxLocBlock(
            matResult,
            nextMaxLoc.minMaxLoc.ptMaxLoc,
            new Size(pTemplData.vecPyramid[iTopLayer].cols, pTemplData.vecPyramid[iTopLayer].rows),
            m_dMaxOverlap,
            blockMax,
          );
          if (nextMaxLoc.minMaxLoc.dMaxVal < vecLayerScore[iTopLayer]) {
            break;
          }
          vecMatchParameter.push({
            pt: new Point2(nextMaxLoc.minMaxLoc.ptMaxLoc.x - fTranslationX, nextMaxLoc.minMaxLoc.ptMaxLoc.y - fTranslationY),
            size: new Size(pTemplData.vecPyramid[iTopLayer].cols, pTemplData.vecPyramid[iTopLayer].rows),
            dMatchScore: nextMaxLoc.minMaxLoc.dMaxVal,
            dMatchAngle: vecAngles[i],
          });
        }
      } else {
        let minMaxLoc = await cv.minMaxLocAsync(matResult);

        if (minMaxLoc.maxVal < vecLayerScore[iTopLayer]) {
          continue;
        }

        vecMatchParameter.push({
          pt: new Point2(minMaxLoc.maxLoc.x - fTranslationX, minMaxLoc.maxLoc.y - fTranslationY),
          dMatchScore: minMaxLoc.maxVal,
          dMatchAngle: vecAngles[i],
          size: { width: matResult.sizes[1], height: matResult.sizes[0] },
        });
        let nextMaxLoc: {
          srcMat: Mat;
          minMaxLoc: {
            minVal: number;
            maxVal: number;
            minLoc: Point2;
            maxLoc: Point2;
          };
        } = { srcMat: matResult, minMaxLoc: minMaxLoc };

        for (let j = 0; j < m_iMaxPos + InvariantRotatingHandler.MATCH_CANDIDATE_NUM - 1; j++) {
          nextMaxLoc = await InvariantRotatingHandler.GetNextMaxLoc(
            nextMaxLoc.srcMat,
            nextMaxLoc.minMaxLoc.maxLoc,
            { width: pTemplData.vecPyramid[iTopLayer].sizes[1], height: pTemplData.vecPyramid[iTopLayer].sizes[0] },
            m_dMaxOverlap,
          );
          if (nextMaxLoc.minMaxLoc.maxVal < vecLayerScore[iTopLayer]) {
            break;
          }
          vecMatchParameter.push({
            pt: new Point2(nextMaxLoc.minMaxLoc.maxLoc.x - fTranslationX, nextMaxLoc.minMaxLoc.maxLoc.y - fTranslationY),
            dMatchScore: nextMaxLoc.minMaxLoc.maxVal,
            dMatchAngle: vecAngles[i],
            size: { width: nextMaxLoc.srcMat.sizes[1], height: nextMaxLoc.srcMat.sizes[0] },
          });
        }
      }
    }
    vecMatchParameter.sort((first, second) => second.dMatchScore - first.dMatchScore);

    let iDstW = pTemplData.vecPyramid[iTopLayer].cols;
    let iDstH = pTemplData.vecPyramid[iTopLayer].rows;

    const iStopLayer = m_bStopLayer1 ? 1 : 0;
    let vecAllResult: Array<MatchParameter> = [];

    for (let i = 0; i < vecMatchParameter.length; i++) {
      const dRAngle = -vecMatchParameter[i].dMatchAngle * InvariantRotatingHandler.D2R;
      let ptLT = InvariantRotatingHandler.ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);

      let dAngleStep = Math.atan(2.0 / Math.max(iDstW, iDstH)) * InvariantRotatingHandler.R2D;
      vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
      vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

      if (iTopLayer <= iStopLayer) {
        vecMatchParameter[i].pt = ptLT.mul(iTopLayer == 0 ? 1 : 2) as Point2;
        vecAllResult.push(vecMatchParameter[i]);
      } else {
        for (let iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--) {
          dAngleStep = Math.atan(2.0 / Math.max(pTemplData.vecPyramid[iLayer].cols, pTemplData.vecPyramid[iLayer].rows)) * InvariantRotatingHandler.R2D; //min改為max
          const vecAngles: Array<number> = [];
          let dMatchedAngle = vecMatchParameter[i].dMatchAngle;

          if (m_bToleranceRange) {
            for (let i = -1; i <= 1; i++) {
              vecAngles.push(dMatchedAngle + dAngleStep * i);
            }
          } else {
            if (m_dToleranceAngle < InvariantRotatingHandler.MAX_VISIBLE) {
              vecAngles.push(0.0);
            } else {
              for (let i = -1; i <= 1; i++) {
                vecAngles.push(dMatchedAngle + dAngleStep * i);
              }
            }
          }
          const ptSrcCenter: Point2 = new Point2((vecMatSrcPyr[iLayer].cols - 1) / 2.0, (vecMatSrcPyr[iLayer].rows - 1) / 2.0);
          const vecNewMatchParameter: Array<MatchParameter> = [];
          let iMaxScoreIndex = 0;
          let dBigValue = -1;

          for (let j = 0; j < vecAngles.length; j++) {
            const matRotatedSrc = await InvariantRotatingHandler.GetRotatedROI(
              vecMatSrcPyr[iLayer],
              new Size(pTemplData.vecPyramid[iLayer].cols, pTemplData.vecPyramid[iLayer].rows),
              ptLT.mul(2) as Point2,
              vecAngles[j],
            );
            const matResult = await InvariantRotatingHandler.MatchTemplate(matRotatedSrc, pTemplData, iLayer);
            const minMax = matResult.minMaxLoc();
            const dMaxValue = minMax.maxVal;
            const ptMaxLoc = minMax.maxLoc;
            vecNewMatchParameter[j] = { pt: ptMaxLoc, dMatchScore: dMaxValue, dMatchAngle: vecAngles[j], size: { width: matResult.cols, height: matResult.rows } };

            if (vecNewMatchParameter[j].dMatchScore > dBigValue) {
              iMaxScoreIndex = j;
              dBigValue = vecNewMatchParameter[j].dMatchScore;
            }
            if (ptMaxLoc.x === 0 || ptMaxLoc.y === 0 || ptMaxLoc.x === matResult.cols - 1 || ptMaxLoc.y === matResult.rows - 1) {
              vecNewMatchParameter[j].bPosOnBorder = true;
            }
            if (!vecNewMatchParameter[j].bPosOnBorder) {
              vecNewMatchParameter[j].vecResult = [[...Array(3).keys()].fill(0), [...Array(3).keys()].fill(0), [...Array(3).keys()].fill(0)];
              for (let y = -1; y <= 1; y++) {
                for (let x = -1; x <= 1; x++) {
                  const ptMaxLocTemp = ptMaxLoc.add(new Point2(x, y)) as Point2;

                  (vecNewMatchParameter[j].vecResult as any)[x + 1][y + 1] = matResult.at(ptMaxLocTemp.y, ptMaxLocTemp.x);
                }
              }
            }
          }
          if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer]) {
            break;
          }
          if (bSubPixelEstimation && iLayer == 0 && !vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder && iMaxScoreIndex != 0 && iMaxScoreIndex != 2) {
            const newCoords = InvariantRotatingHandler.SubPixEstimation(vecNewMatchParameter, dAngleStep, iMaxScoreIndex);
            vecNewMatchParameter[iMaxScoreIndex].pt = new Point2(newCoords.dNewX, newCoords.dNewY);
            vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = newCoords.dNewAngle;
          }

          const dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;
          const ptPaddingLT: Point2 = InvariantRotatingHandler.ptRotatePt2f(ptLT.mul(2) as Point2, ptSrcCenter, dNewMatchAngle * InvariantRotatingHandler.D2R).sub(new Point2(3, 3)) as Point2;
          let pt = new Point2(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
          pt = InvariantRotatingHandler.ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * InvariantRotatingHandler.D2R);

          if (iLayer == iStopLayer) {
            vecNewMatchParameter[iMaxScoreIndex].pt = pt.mul(iStopLayer == 0 ? 1 : 2) as Point2;
            vecAllResult.push(vecNewMatchParameter[iMaxScoreIndex]);
          } else {
            vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
            vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep / 2;
            vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep / 2;
            ptLT = pt;
          }
        }
      }
    }
    vecAllResult = vecAllResult.filter((f) => f.dMatchScore >= dScore);

    iDstW = pTemplData.vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
    iDstH = pTemplData.vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

    for (let i = 0; i < vecAllResult.length; i++) {
      let ptLT;
      let ptRT;
      let ptRB;
      let ptLB;

      const dRAngle = -vecAllResult[i].dMatchAngle * InvariantRotatingHandler.D2R;

      ptLT = vecAllResult[i].pt;
      ptRT = new Point2(ptLT.x + iDstW * Math.cos(dRAngle), ptLT.y - iDstW * Math.sin(dRAngle));
      ptLB = new Point2(ptLT.x + iDstH * Math.sin(dRAngle), ptLT.y + iDstH * Math.cos(dRAngle));
      ptRB = new Point2(ptRT.x + iDstH * Math.sin(dRAngle), ptRT.y + iDstH * Math.cos(dRAngle));
      const rotatedRectPoint = InvariantRotatingHandler.RotatedRect(ptLT, ptRT, ptRB);
      vecAllResult[i].rectR = new RotatedRect(rotatedRectPoint.center, rotatedRectPoint.size, rotatedRectPoint.angle);
    }
    vecAllResult = InvariantRotatingHandler.FilterWithRotatedRect(vecAllResult, 'CV_TM_CCOEFF_NORMED', m_dMaxOverlap);

    const iW = pTemplData.vecPyramid[0].cols;
    const iH = pTemplData.vecPyramid[0].rows;

    let m_vecSingleTargetData: Array<SingleTargetMatch> = [];

    for (let i = 0; i < vecAllResult.length; i++) {
      let sstm: SingleTargetMatch = {
        ptLT: new Point2(0, 0),
        ptRT: new Point2(0, 0),
        ptRB: new Point2(0, 0),
        ptLB: new Point2(0, 0),
        ptCenter: new Point2(0, 0),
        dMatchedAngle: 0,
        dMatchScore: 0,
        size: { width: 0, height: 0 },
      };
      const dRAngle = -vecAllResult[i].dMatchAngle * InvariantRotatingHandler.D2R;

      sstm.ptLT = vecAllResult[i].pt;
      sstm.ptRT = new Point2(sstm.ptLT.x + iW * Math.cos(dRAngle), sstm.ptLT.y - iW * Math.sin(dRAngle));
      sstm.ptLB = new Point2(sstm.ptLT.x + iH * Math.sin(dRAngle), sstm.ptLT.y + iH * Math.cos(dRAngle));
      sstm.ptRB = new Point2(sstm.ptRT.x + iH * Math.sin(dRAngle), sstm.ptRT.y + iH * Math.cos(dRAngle));
      sstm.ptCenter = new Point2((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
      sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
      sstm.dMatchScore = vecAllResult[i].dMatchScore;
      sstm.size = vecAllResult[i].rectR?.size as Size;
      if (sstm.dMatchedAngle < -180) {
        sstm.dMatchedAngle += 360;
      }
      if (sstm.dMatchedAngle > 180) {
        sstm.dMatchedAngle -= 360;
      }
      m_vecSingleTargetData.push(sstm);

      if (i + 1 == m_iMaxPos) {
        break;
      }
    }

    if (debug) {
      for (let sstm of m_vecSingleTargetData) {
        let ptDis1;
        let ptDis2;

        if (matDst.cols > matDst.rows) {
          ptDis1 = sstm.ptLB.sub(sstm.ptLT).div(3);
          ptDis2 = sstm.ptRT
            .sub(sstm.ptLT)
            .div(3)
            .mul(matDst.rows / matDst.cols);
        } else {
          ptDis1 = sstm.ptLB
            .sub(sstm.ptLT)
            .div(3)
            .mul(matDst.cols / matDst.rows);
          ptDis2 = sstm.ptRT.sub(sstm.ptLT).div(3);
        }
        matSrc.drawLine(sstm.ptLT, sstm.ptLT.add(ptDis1.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);
        matSrc.drawLine(sstm.ptLT, sstm.ptLT.add(ptDis2.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);
        matSrc.drawLine(sstm.ptRT, sstm.ptRT.add(ptDis1.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);
        matSrc.drawLine(sstm.ptRT, sstm.ptRT.sub(ptDis2.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);
        matSrc.drawLine(sstm.ptRB, sstm.ptRB.sub(ptDis1.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);
        matSrc.drawLine(sstm.ptRB, sstm.ptRB.sub(ptDis2.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);
        matSrc.drawLine(sstm.ptLB, sstm.ptLB.sub(ptDis1.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);
        matSrc.drawLine(sstm.ptLB, sstm.ptLB.add(ptDis2.div(2)) as Point2, new Vec3(0, 255, 0), 1, 16);

        matSrc.drawLine(sstm.ptCenter?.sub(new Point2(5, 0)) as Point2, sstm.ptCenter?.add(new Point2(5, 0)) as Point2, new Vec3(255, 0, 0), 2);
        matSrc.drawLine(sstm.ptCenter?.sub(new Point2(0, 5)) as Point2, sstm.ptCenter?.add(new Point2(0, 5)) as Point2, new Vec3(255, 0, 0), 2);
      }

      const endTimeCheck = performance.now();
      console.log(`Performing time was ${endTimeCheck - startTimeCheck}`);

      cv.imshow('debug', matSrc);
      cv.waitKey(0);
    }
    return m_vecSingleTargetData;
  }

  private static SubPixEstimation(vec: Array<MatchParameter>, dAngleStep: number, iMaxScoreIndex: number) {
    let matA = new Mat(27, 10, CV_64F);
    let matZ = new Mat(10, 1, CV_64F);
    let matS = new Mat(27, 1, CV_64F);

    const dX_maxScore = vec[iMaxScoreIndex].pt.x;
    const dY_maxScore = vec[iMaxScoreIndex].pt.y;
    const dTheata_maxScore = vec[iMaxScoreIndex].dMatchAngle;
    let iRow = 0;

    for (let theta = 0; theta <= 2; theta++) {
      for (let y = -1; y <= 1; y++) {
        for (let x = -1; x <= 1; x++) {
          const dX = dX_maxScore + x;
          const dY = dY_maxScore + y;

          const dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * InvariantRotatingHandler.D2R;
          matA.set(iRow, 0, dX * dX);
          matA.set(iRow, 1, dY * dY);
          matA.set(iRow, 2, dT * dT);
          matA.set(iRow, 3, dX * dY);
          matA.set(iRow, 4, dX * dT);
          matA.set(iRow, 5, dY * dT);
          matA.set(iRow, 6, dX);
          matA.set(iRow, 7, dY);
          matA.set(iRow, 8, dT);
          matA.set(iRow, 9, 1.0);
          matS.set(iRow, 0, (vec[iMaxScoreIndex + (theta - 1)] as any).vecResult[x + 1][y + 1]);
          iRow++;
        }
      }
    }
    const matATransposed = InvariantRotatingHandler.Transpose(matA);
    const matAMultiplicated = matATransposed.matMul(matA);
    const matAInverted = cv.invert(matAMultiplicated);
    const matAInvertedMultiplicated = matAInverted.matMul(InvariantRotatingHandler.Transpose(matA));
    matZ = matAInvertedMultiplicated.matMul(matS);
    const matZ_t = InvariantRotatingHandler.Transpose(matZ);

    let matK1 = new Mat(3, 3, cv.CV_64F);

    matK1.set(0, 0, 2 * matZ_t.at(0, 0));
    matK1.set(0, 1, matZ_t.at(0, 3));
    matK1.set(0, 2, matZ_t.at(0, 4));
    matK1.set(1, 0, matZ_t.at(0, 3));
    matK1.set(1, 1, 2 * matZ_t.at(0, 1));
    matK1.set(1, 2, matZ_t.at(0, 5));
    matK1.set(2, 0, matZ_t.at(0, 4));
    matK1.set(2, 1, matZ_t.at(0, 5));
    matK1.set(2, 2, 2 * matZ_t.at(0, 2));

    let matK2 = new Mat(3, 1, cv.CV_64F);

    matK2.set(0, 0, -matZ_t.at(0, 6));
    matK2.set(1, 0, -matZ_t.at(0, 7));
    matK2.set(2, 0, -matZ_t.at(0, 8));

    const matDelta = cv.invert(matK1).matMul(matK2);

    const dNewX = matDelta.at(0, 0);
    const dNewY = matDelta.at(1, 0);
    const dNewAngle = matDelta.at(2, 0) * InvariantRotatingHandler.R2D;

    return { dNewX: dNewX, dNewY: dNewY, dNewAngle: dNewAngle };
  }

  static Transpose(mat: Mat) {
    const data = mat.getDataAsArray();
    const transposed: number[][] = Array.from(new Array(mat.cols), () => []);

    for (let k = 0; k < mat.cols; k++) {
      transposed[k] = [...Array(mat.rows).keys()].fill(0);
    }

    for (let j = 0; j < mat.cols; j++) {
      for (let i = 0; i < mat.rows; i++) {
        transposed[j][i] = data[i][j];
      }
    }

    return new Mat(transposed, CV_64F);
  }

  static Invert(image: Mat) {
    const data = image.getData();

    for (let i = 0; i < data.length; i += 8) {
      data[i] = 255 - data[i];
      data[i + 1] = Math.abs(255 - data[i + 1]);
      data[i + 2] = Math.abs(255 - data[i + 2]);
      data[i + 3] = Math.abs(255 - data[i + 3]);
      data[i + 4] = Math.abs(255 - data[i + 4]);
      data[i + 5] = Math.abs(255 - data[i + 5]);
      data[i + 6] = Math.abs(255 - data[i + 6]);
      data[i + 7] = Math.abs(255 - data[i + 7]);
    }
    return new Mat(data, image.rows, image.cols, CV_64F);
  }

  private static RotatedRect(ptLT: Point2, ptRT: Point2, ptRB: Point2) {
    const cx = (ptLT.x + ptRT.x + ptRB.x) / 3;
    const cy = (ptLT.y + ptRT.y + ptRB.y) / 3;

    const width = Math.sqrt(Math.pow(ptRT.x - ptLT.x, 2) + Math.pow(ptRT.y - ptLT.y, 2));
    const height = Math.sqrt(Math.pow(ptRB.x - ptRT.x, 2) + Math.pow(ptRB.y - ptRT.y, 2));
    const angle = Math.atan2(ptRT.y - ptLT.y, ptRT.x - ptLT.x) * InvariantRotatingHandler.R2D;

    return {
      center: new Point2(cx, cy),
      size: new Size(width, height),
      angle: angle,
    };
  }

  private static FilterWithRotatedRect(vec: Array<MatchParameter>, iMethod: 'CV_TM_SQDIFF' | 'CV_TM_CCOEFF_NORMED', dMaxOverLap: number) {
    const iMatchSize = vec.length;
    for (let i = 0; i < iMatchSize - 1; i++) {
      if (vec[i].bDelete) {
        continue;
      }
      for (let j = i + 1; j < iMatchSize; j++) {
        if (vec[j].bDelete) {
          continue;
        }
        let rect1: RotatedRect = vec[i].rectR as RotatedRect;
        let rect2: RotatedRect = vec[j].rectR as RotatedRect;

        const iInterSec = InvariantRotatingHandler.rotatedRectangleIntersection(rect1, rect2);

        if (iInterSec.intersectionType === InvariantRotatingHandler.INTERSECT_NONE) {
          continue;
        } else if (iInterSec.intersectionType === InvariantRotatingHandler.INTERSECT_FULL) {
          let iDeleteIndex;
          if (iMethod == 'CV_TM_SQDIFF') {
            iDeleteIndex = vec[i].dMatchScore <= vec[j].dMatchScore ? j : i;
          } else {
            iDeleteIndex = vec[i].dMatchScore >= vec[j].dMatchScore ? j : i;
          }
          vec[iDeleteIndex].bDelete = true;
        } else {
          if (iInterSec.points.length < 3) {
            continue;
          } else {
            let iDeleteIndex;

            InvariantRotatingHandler.sortPtWithCenter(iInterSec.points);
            const dArea = InvariantRotatingHandler.contourArea(iInterSec.points);
            const dRatio = dArea / (rect1.size.height * rect1.size.width);

            if (dRatio > dMaxOverLap) {
              if (iMethod == 'CV_TM_SQDIFF') {
                iDeleteIndex = vec[i].dMatchScore <= vec[j].dMatchScore ? j : i;
              } else {
                iDeleteIndex = vec[i].dMatchScore >= vec[j].dMatchScore ? j : i;
              }
              vec[iDeleteIndex].bDelete = true;
            }
          }
        }
      }
    }
    for (let i = 0; i < vec.length; ) {
      if (vec[i].bDelete) {
        vec.splice(i, 1);
      } else {
        i++;
      }
    }
    return vec;
  }

  private static rotatedRectangleIntersection(rect1: RotatedRect, rect2: RotatedRect) {
    return rotatedRectangle(rect1, rect2);

    function rotatedRectangle(rect1: RotatedRect, rect2: RotatedRect) {
      let rect1Corners = getCorners(rect1);
      let rect2Corners = getCorners(rect2);
      let rect1InsideRect2 = true;
      let rect2InsideRect1 = true;
      let intersectionPoints = [];

      for (let i = 0; i < rect1Corners.length; i++) {
        if (isPointInside(rect1Corners[i], rect2Corners)) {
          intersectionPoints.push(rect1Corners[i]);
        } else {
          rect1InsideRect2 = false;
        }
      }
      for (let i = 0; i < rect2Corners.length; i++) {
        if (isPointInside(rect2Corners[i], rect1Corners)) {
          intersectionPoints.push(rect2Corners[i]);
        } else {
          rect2InsideRect1 = false;
        }
      }
      for (let i = 0; i < rect1Corners.length; i++) {
        let j = (i + 1) % rect1Corners.length;
        for (let k = 0; k < rect2Corners.length; k++) {
          let l = (k + 1) % rect2Corners.length;
          let intersectionPoint = doLinesIntersect(rect1Corners[i], rect1Corners[j], rect2Corners[k], rect2Corners[l]);
          if (intersectionPoint) {
            intersectionPoints.push(intersectionPoint);
          }
        }
      }

      if (intersectionPoints.length === 0) {
        return { intersectionType: 0, points: [] };
      }

      if (rect1InsideRect2) {
        return { intersectionType: 2, points: rect1Corners };
      }

      if (rect2InsideRect1) {
        return { intersectionType: 2, points: rect2Corners };
      }
      return { intersectionType: 1, points: intersectionPoints };
    }

    function getCorners(rect: RotatedRect) {
      let x = rect.center.x;
      let y = rect.center.y;
      let w = rect.size.width;
      let h = rect.size.height;
      let angle = rect.angle;

      let cos = Math.cos(angle);
      let sin = Math.sin(angle);

      let topLeft = { x: x + (-w / 2) * cos - (-h / 2) * sin, y: y + (-w / 2) * sin + (-h / 2) * cos };
      let topRight = { x: x + (w / 2) * cos - (-h / 2) * sin, y: y + (w / 2) * sin + (-h / 2) * cos };
      let bottomRight = { x: x + (w / 2) * cos - (h / 2) * sin, y: y + (w / 2) * sin + (h / 2) * cos };
      let bottomLeft = { x: x + (-w / 2) * cos - (h / 2) * sin, y: y + (-w / 2) * sin + (h / 2) * cos };

      return [new Point2(topLeft.x, topLeft.y), new Point2(topRight.x, topRight.y), new Point2(bottomRight.x, bottomRight.y), new Point2(bottomLeft.x, bottomLeft.y)];
    }

    function isPointInside(point: Point2, polygon: Array<Point2>) {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        let xi = polygon[i].x;
        let yi = polygon[i].y;
        let xj = polygon[j].x;
        let yj = polygon[j].y;
        let intersect = yi > point.y != yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
        if (intersect) {
          inside = !inside;
        }
      }
      return inside;
    }

    function doLinesIntersect(a: Point2, b: Point2, c: Point2, d: Point2) {
      let denominator = (d.y - c.y) * (b.x - a.x) - (d.x - c.x) * (b.y - a.y);
      if (denominator == 0) {
        return false;
      }
      let ua = ((d.x - c.x) * (a.y - c.y) - (d.y - c.y) * (a.x - c.x)) / denominator;
      let ub = ((b.x - a.x) * (a.y - c.y) - (b.y - a.y) * (a.x - c.x)) / denominator;
      if (ua < 0 || ua > 1 || ub < 0 || ub > 1) {
        return false;
      }
      let intersectionX = a.x + ua * (b.x - a.x);
      let intersectionY = a.y + ua * (b.y - a.y);

      return new Point2(intersectionX, intersectionY);
    }
  }

  private static contourArea(contour: Array<any>, oriented = false) {
    let area = 0;
    const n = contour.length;

    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += contour[i].x * contour[j].y - contour[j].x * contour[i].y;
    }

    area /= 2;

    if (oriented) {
      return area;
    } else {
      return Math.abs(area);
    }
  }

  private static sortPtWithCenter(vecSort: Array<Point2>) {
    const iSize = vecSort.length;
    let ptCenter = { x: 0, y: 0 };

    for (let i = 0; i < iSize; i++) {
      ptCenter.x += vecSort[i].x;
      ptCenter.y += vecSort[i].y;
    }

    ptCenter.x /= iSize;
    ptCenter.y /= iSize;

    const vecPtAngle = [];

    for (let i = 0; i < iSize; i++) {
      const vec1 = {
        x: vecSort[i].x - ptCenter.x,
        y: vecSort[i].y - ptCenter.y,
      };

      const fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
      const fDot = vec1.x;

      let angle;

      if (vec1.y < 0) {
        angle = Math.acos(fDot / fNormVec1) * InvariantRotatingHandler.R2D;
      } else if (vec1.y > 0) {
        angle = 360 - Math.acos(fDot / fNormVec1) * InvariantRotatingHandler.R2D;
      } else {
        if (vec1.x - ptCenter.x > 0) {
          angle = 0;
        } else {
          angle = 180;
        }
      }

      vecPtAngle.push({ point: vecSort[i], angle });
    }

    vecPtAngle.sort((a, b) => a.angle - b.angle);

    for (let i = 0; i < iSize; i++) {
      vecSort[i] = vecPtAngle[i].point;
    }
  }

  private static async GetNextMaxLoc(matResult: Mat, ptMaxLoc: Point2, sizeTemplate: Size, dMaxOverlap: number) {
    const iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
    const iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);

    matResult.drawRectangle(new Rect(iStartX, iStartY, 2 * sizeTemplate.width * (1 - dMaxOverlap), 2 * sizeTemplate.height * (1 - dMaxOverlap)), new Vec3(-1, -1, -1), FILLED);

    const ptNewMaxLoc = await cv.minMaxLocAsync(matResult);

    return { srcMat: matResult, minMaxLoc: ptNewMaxLoc };
  }

  private static async GetNextMaxLocBlock(matResult: Mat, ptMaxLoc: Point2, sizeTemplate: Size, dMaxOverlap: number, blockMax: SBlockMax) {
    let iStartX = Math.floor(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
    let iStartY = Math.floor(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
    const rectIgnore = new Rect(iStartX, iStartY, Math.floor(2 * sizeTemplate.width * (1 - dMaxOverlap)), Math.floor(2 * sizeTemplate.height * (1 - dMaxOverlap)));
    matResult.drawRectangle(rectIgnore, new Vec3(-1, -1, -1), FILLED);
    blockMax.UpdateMax(rectIgnore);
    const ptReturn = blockMax.GetMaxValueLoc();

    return {
      srcMat: matResult,
      minMaxLoc: {
        dMaxVal: ptReturn.dMaxVal,
        ptMaxLoc: ptReturn.ptMaxLoc,
      },
    };
  }

  private static readDoubleLE(array: Array<number>, offset: number = 0) {
    const int1 = array[offset] | (array[offset + 1] << 8) | (array[offset + 2] << 16) | (array[offset + 3] << 24);
    const int2 = array[offset + 4] | (array[offset + 5] << 8) | (array[offset + 6] << 16) | (array[offset + 7] << 24);
    const sign = int2 & 0x80000000 ? -1 : 1;
    const exponent = ((int2 >>> 20) & 0x7ff) - 1023;
    const mantissa = (int2 & 0xfffff) * 0x100000000 + int1;

    if (exponent === 1024) {
      return mantissa ? NaN : sign * Infinity;
    }

    if (exponent === -1023) {
      return sign * mantissa * 1.1102230246251565e-16;
    }

    return sign * (1 + mantissa * 2.220446049250313e-16) * Math.pow(2, exponent);
  }

  private static getArrayFromBuffer(buffer: Buffer, size: Size, bites: number = 8, offset: number, makePixelTransformation: (arr: Array<number>, offset: number) => number) {
    const transformedImageData: any = Array.from(new Array(size.height), () => []);

    for (let i = 1; i <= buffer.length; i += bites) {
      let pixel: Array<number> = [];
      const row = Math.floor(i / bites / size.width);

      for (let k = 0; k < bites - 1; k++) {
        if (k === 0) {
          pixel.push(buffer[i - 1]);
        }
        pixel.push(buffer[i + k]);
      }
      transformedImageData[row].push(makePixelTransformation(pixel, offset));
    }
    return transformedImageData;
  }

  private static getArrayFromUintBuffer(mat: Mat, bytes: '32' | '64') {
    let buffer;
    let uint = new Uint8Array(mat.getData());

    if (bytes === '32') {
      buffer = new Float32Array(uint.buffer);
    } else {
      buffer = new Float64Array(uint.buffer);
    }
    const arr: number[][] = Array.from(new Array(mat.rows), () => []);

    for (let i = 0; i < buffer.length; i++) {
      const row = Math.floor(i / mat.cols);

      arr[row].push(buffer[i]);
    }
    return arr;
  }

  private static async MatchTemplate(matSrc: Mat, pTemplData: Vector, iLayer: number, method: 'TM_CCOEFF_NORMED' | 'TM_CCORR' = 'TM_CCORR') {
    const result = await matSrc.matchTemplateAsync(pTemplData.vecPyramid[iLayer], cv[method]);

    return await InvariantRotatingHandler.CCOEFF_Denominator(matSrc, pTemplData, result, iLayer);
  }

  private static async CCOEFF_Denominator(matSrc: Mat, pTemplData: Vector, matResult: Mat, iLayer: number) {
    if (pTemplData.vecResultEqual1[iLayer]) {
      await matResult.setToAsync(new Vec3(1, 1, 1));

      return matResult;
    }
    let sum: Mat;
    let sqsum: Mat;

    const integral = await matSrc.integralAsync(CV_64F);

    sum = integral.sum;
    sqsum = integral.sqsum;

    const sumArr = InvariantRotatingHandler.getArrayFromUintBuffer(sum, '64');
    const sqsumArr = InvariantRotatingHandler.getArrayFromUintBuffer(sqsum, '64');
    const matResultArr = InvariantRotatingHandler.getArrayFromUintBuffer(matResult, '32');

    // const sumArr = InvariantRotatingHandler.getArrayFromBuffer(sum.getData(), { width: sum.cols, height: sum.rows }, 8, 0, InvariantRotatingHandler.readDoubleLE);
    // const sqsumArr = InvariantRotatingHandler.getArrayFromBuffer(sqsum.getData(), { width: sum.cols, height: sum.rows }, 8, 0, InvariantRotatingHandler.readDoubleLE);

    // const sumArr = sum.getDataAsArray();
    // const sqsumArr = sqsum.getDataAsArray();

    const q0 = { rows: 0, cols: 0 };
    const q1 = { rows: 0, cols: pTemplData.vecPyramid[iLayer].cols };
    const q2 = { rows: pTemplData.vecPyramid[iLayer].rows, cols: 0 };
    const q3 = { rows: pTemplData.vecPyramid[iLayer].rows, cols: pTemplData.vecPyramid[iLayer].cols };

    const dTemplMean0 = pTemplData.vecTemplMean[iLayer][0];
    const dTemplNorm = pTemplData.vecTemplNorm[iLayer];
    const dInvArea = pTemplData.vecInvArea[iLayer];

    for (let i = 0; i < matResult.rows; i++) {
      for (let j = 0; j < matResult.cols; j += 1) {
        let num = matResult.at(i, j);
        let t;
        let wndMean2 = 0;
        let wndSum2 = 0;

        t = sumArr[q0.rows + i][q0.cols + j] - sumArr[q1.rows + i][q1.cols + j] - sumArr[q2.rows + i][q2.cols + j] + sumArr[q3.rows + i][q3.cols + j];
        wndMean2 += t * t;
        num -= t * dTemplMean0;
        wndMean2 *= dInvArea;

        t = sqsumArr[q0.rows + i][q0.cols + j] - sqsumArr[q1.rows + i][q1.cols + j] - sqsumArr[q2.rows + i][q2.cols + j] + sqsumArr[q3.rows + i][q3.cols + j];
        wndSum2 += t;

        let diff2 = Math.max(wndSum2 - wndMean2, 0);

        if (diff2 <= Math.min(0.5, 10 * 1.19209e-7 * wndSum2)) {
          t = 0;
        } else {
          t = Math.sqrt(diff2) * dTemplNorm;
        }

        if (Math.abs(num) < t) {
          num /= t;
        } else if (Math.abs(num) < t * 1.125) {
          num = num > 0 ? 1 : -1;
        } else {
          num = 0;
        }
        // matResult.set(i, j, num);
        matResultArr[i][j] = num;
      }
    }
    // return matResult;
    return new Mat(matResultArr, CV_32F);
  }

  private static async LearnPattern(matDst: Mat, iMinDstLength: number) {
    let templData: Vector = { vecInvArea: [], vecPyramid: [], vecTemplMean: [], vecTemplNorm: [], vecResultEqual1: [], bIsPatternLearned: true, iBorderColor: 0 };
    let iTopLayer = InvariantRotatingHandler.GetTopLayer(matDst, Math.sqrt(iMinDstLength));
    const pyramid = await matDst.buildPyramidAsync(iTopLayer);
    templData.vecPyramid = pyramid;
    templData.iBorderColor = cv.mean(matDst).w < 128 ? 255 : 0;

    for (let i = 0; i < templData.vecPyramid.length; i++) {
      const invArea = 1 / (templData.vecPyramid[i].rows * templData.vecPyramid[i].cols);

      let templMeanAndSdv = cv.meanStdDev(templData.vecPyramid[i]);
      let templNorm = templMeanAndSdv.stddev.at(0, 0) * templMeanAndSdv.stddev.at(0, 0);

      if (templNorm < Number.EPSILON) {
        templData.vecResultEqual1[i] = true;
      }
      let templSum2 = templNorm + templMeanAndSdv.mean.at(0, 0) * templMeanAndSdv.mean.at(0, 0);

      templSum2 /= invArea;
      templNorm = Math.sqrt(templNorm);
      templNorm /= Math.sqrt(invArea);

      templData.vecInvArea[i] = invArea;
      templData.vecTemplMean[i] = [templMeanAndSdv.mean.at(0, 0), 0, 0, 0];
      templData.vecTemplNorm[i] = templNorm;
    }
    templData.bIsPatternLearned = true;

    return templData;
  }

  private static GetBestRotationSize(sizeSrc: { width: number; height: number }, sizeDst: { width: number; height: number }, dRAngle: number): { width: number; height: number } {
    const ptLT = { x: 0, y: 0 };
    const ptLB = { x: 0, y: sizeSrc.height - 1 };
    const ptRB = { x: sizeSrc.width - 1, y: sizeSrc.height - 1 };
    const ptRT = { x: sizeSrc.width - 1, y: 0 };
    const ptCenter = new Point2((sizeSrc.width - 1) / 2.0, (sizeSrc.height - 1) / 2.0);
    const ptLT_R = InvariantRotatingHandler.ptRotatePt2f(new Point2(ptLT.x, ptLT.y), ptCenter, dRAngle * InvariantRotatingHandler.D2R);
    const ptLB_R = InvariantRotatingHandler.ptRotatePt2f(new Point2(ptLB.x, ptLB.y), ptCenter, dRAngle * InvariantRotatingHandler.D2R);
    const ptRB_R = InvariantRotatingHandler.ptRotatePt2f(new Point2(ptRB.x, ptRB.y), ptCenter, dRAngle * InvariantRotatingHandler.D2R);
    const ptRT_R = InvariantRotatingHandler.ptRotatePt2f(new Point2(ptRT.x, ptRT.y), ptCenter, dRAngle * InvariantRotatingHandler.D2R);
    const fTopY = Math.max(ptLT_R.y, ptLB_R.y, ptRB_R.y, ptRT_R.y);
    const fBottomY = Math.min(ptLT_R.y, ptLB_R.y, ptRB_R.y, ptRT_R.y);
    const fRightX = Math.max(ptLT_R.x, ptLB_R.x, ptRB_R.x, ptRT_R.x);
    const fLeftX = Math.min(ptLT_R.x, ptLB_R.x, ptRB_R.x, ptRT_R.x);

    if (dRAngle > 360) {
      dRAngle -= 360;
    } else if (dRAngle < 0) {
      dRAngle += 360;
    }

    if (Math.abs(Math.abs(dRAngle) - 90) < InvariantRotatingHandler.MAX_VISIBLE || Math.abs(Math.abs(dRAngle) - 270) < InvariantRotatingHandler.MAX_VISIBLE) {
      return { width: sizeSrc.height, height: sizeSrc.width };
    } else if (Math.abs(dRAngle) < InvariantRotatingHandler.MAX_VISIBLE || Math.abs(Math.abs(dRAngle) - 180) < InvariantRotatingHandler.MAX_VISIBLE) {
      return sizeSrc;
    }

    let dAngle = dRAngle;

    if (dAngle > 0 && dAngle < 90) {
      // do nothing
    } else if (dAngle > 90 && dAngle < 180) {
      dAngle -= 90;
    } else if (dAngle > 180 && dAngle < 270) {
      dAngle -= 180;
    } else if (dAngle > 270 && dAngle < 360) {
      dAngle -= 270;
    }

    const fH1 = sizeDst.width * Math.sin(dAngle * InvariantRotatingHandler.D2R) * Math.cos(dAngle * InvariantRotatingHandler.D2R);
    const fH2 = sizeDst.height * Math.sin(dAngle * InvariantRotatingHandler.D2R) * Math.cos(dAngle * InvariantRotatingHandler.D2R);
    const iHalfHeight = Math.ceil(fTopY - ptCenter.y - fH1);
    const iHalfWidth = Math.ceil(fRightX - ptCenter.x - fH2);
    let sizeRet = { width: iHalfWidth * 2, height: iHalfHeight * 2 };
    const bWrongSize =
      (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height) ||
      (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height) ||
      sizeDst.width * sizeDst.height > sizeRet.width * sizeRet.width;

    if (bWrongSize) {
      sizeRet = { width: Math.round(fRightX - fLeftX), height: Math.round(fTopY - fBottomY) };
    }
    return sizeRet;
  }

  private static ptRotatePt2f(ptInput: Point2, ptOrg: Point2, dAngle: number): Point2 {
    let dHeight = ptOrg.y * 2;
    let dY1 = dHeight - ptInput.y;
    let dY2 = dHeight - ptOrg.y;

    let dX = (ptInput.x - ptOrg.x) * Math.cos(dAngle) - (dY1 - ptOrg.y) * Math.sin(dAngle) + ptOrg.x;
    let dY = (ptInput.x - ptOrg.x) * Math.sin(dAngle) + (dY1 - ptOrg.y) * Math.cos(dAngle) + dY2;

    dY = -dY + dHeight;

    return new Point2(dX, dY);
  }
}

interface Block {
  rect: Rect;
  dMax: number;
  ptMaxLoc: Point2;
}

class SBlockMax {
  vecBlock: Array<Block> = [];

  constructor(readonly matSrc: Mat, readonly sizeTemplate: Size) {
    const iBlockW = sizeTemplate.width * 2;
    const iBlockH = sizeTemplate.height * 2;

    const iCol = Math.floor(matSrc.cols / iBlockW);
    const bHResidue = matSrc.cols % iBlockW != 0;

    const iRow = Math.floor(matSrc.rows / iBlockH);
    const bVResidue = matSrc.rows % iBlockH != 0;

    if (iCol === 0 || iRow === 0) {
      this.vecBlock = [];
      return;
    }
    for (let i = 0; i < iCol * iRow; i++) {
      this.vecBlock.push({ rect: new Rect(), dMax: 0, ptMaxLoc: new Point2(0, 0) });
    }

    let iCount = 0;

    for (let y = 0; y < iRow; y++) {
      for (let x = 0; x < iCol; x++) {
        const rectBlock = new Rect(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
        this.vecBlock[iCount].rect = rectBlock;
        const minMax = cv.minMaxLoc(matSrc.getRegion(rectBlock));
        this.vecBlock[iCount].dMax = minMax.maxVal;
        this.vecBlock[iCount].ptMaxLoc = minMax.maxLoc;
        this.vecBlock[iCount].ptMaxLoc = this.vecBlock[iCount].ptMaxLoc.add(new Point2(rectBlock.x, rectBlock.y)) as Point2;

        iCount++;
      }
    }
    if (bHResidue && bVResidue) {
      const rectRight = new Rect(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
      const maxMin = cv.minMaxLoc(matSrc.getRegion(rectRight));
      let blockRight: Block = { dMax: 0, rect: new Rect(), ptMaxLoc: new Point2(0, 0) };
      blockRight.dMax = maxMin.maxVal;
      blockRight.rect = rectRight;
      blockRight.ptMaxLoc = maxMin.maxLoc;
      blockRight.ptMaxLoc = blockRight.ptMaxLoc.add(new Point2(rectRight.x, rectRight.y)) as Point2;
      this.vecBlock.push(blockRight);

      const rectBottom = new Rect(0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
      const maxMin2 = cv.minMaxLoc(matSrc.getRegion(rectBottom));
      let blockBottom: Block = { dMax: 0, rect: new Rect(), ptMaxLoc: new Point2(0, 0) };
      blockBottom.dMax = maxMin2.maxVal;
      blockBottom.rect = rectBottom;
      blockBottom.ptMaxLoc = maxMin2.maxLoc;
      blockBottom.ptMaxLoc = blockBottom.ptMaxLoc.add(new Point2(rectBottom.x, rectBottom.y)) as Point2;
      this.vecBlock.push(blockBottom);
    } else if (bHResidue) {
      const rectRight = new Rect(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
      const maxMin = cv.minMaxLoc(matSrc.getRegion(rectRight));
      let blockRight: Block = { dMax: 0, rect: new Rect(), ptMaxLoc: new Point2(0, 0) };
      blockRight.dMax = maxMin.maxVal;
      blockRight.rect = rectRight;
      blockRight.ptMaxLoc = maxMin.maxLoc;
      blockRight.ptMaxLoc = blockRight.ptMaxLoc.add(new Point2(rectRight.x, rectRight.y)) as Point2;
      this.vecBlock.push(blockRight);
    } else {
      const rectBottom = new Rect(0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
      const maxMin2 = cv.minMaxLoc(matSrc.getRegion(rectBottom));
      let blockBottom: Block = { dMax: 0, rect: new Rect(), ptMaxLoc: new Point2(0, 0) };
      blockBottom.dMax = maxMin2.maxVal;
      blockBottom.rect = rectBottom;
      blockBottom.ptMaxLoc = maxMin2.maxLoc;
      blockBottom.ptMaxLoc = blockBottom.ptMaxLoc.add(new Point2(rectBottom.x, rectBottom.y)) as Point2;
      this.vecBlock.push(blockBottom);
    }
  }
  UpdateMax(rectIgnore: Rect) {
    if (this.vecBlock.length == 0) {
      return;
    }

    const iSize = this.vecBlock.length;

    for (let i = 0; i < iSize; i++) {
      if (
        !(
          this.vecBlock[i].rect.x < rectIgnore.x + rectIgnore.width &&
          this.vecBlock[i].rect.x + this.vecBlock[i].rect.width > rectIgnore.x &&
          this.vecBlock[i].rect.y < rectIgnore.y + rectIgnore.height &&
          this.vecBlock[i].rect.y + this.vecBlock[i].rect.height > rectIgnore.y
        )
      ) {
        continue;
      }

      const minMax = cv.minMaxLoc(this.matSrc.getRegion(this.vecBlock[i].rect));
      this.vecBlock[i].dMax = minMax.maxVal;
      this.vecBlock[i].ptMaxLoc = minMax.maxLoc;
      this.vecBlock[i].ptMaxLoc = this.vecBlock[i].ptMaxLoc.add(new Point2(this.vecBlock[i].rect.x, this.vecBlock[i].rect.y)) as Point2;
    }
  }
  GetMaxValueLoc() {
    const iSize = this.vecBlock.length;
    let dMax, ptMaxLoc;

    if (iSize == 0) {
      const maxMin = cv.minMaxLoc(this.matSrc);
      dMax = maxMin.maxVal;
      ptMaxLoc = maxMin.maxLoc;

      return { dMaxVal: dMax, ptMaxLoc: ptMaxLoc };
    }
    let iIndex = 0;
    dMax = this.vecBlock[0].dMax;
    for (let i = 1; i < iSize; i++) {
      if (this.vecBlock[i].dMax >= dMax) {
        iIndex = i;
        dMax = this.vecBlock[i].dMax;
      }
    }
    ptMaxLoc = this.vecBlock[iIndex].ptMaxLoc;

    return { dMaxVal: dMax, ptMaxLoc: ptMaxLoc };
  }
}
