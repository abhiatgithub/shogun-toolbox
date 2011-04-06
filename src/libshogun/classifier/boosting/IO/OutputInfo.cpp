/*
 *
 *    MultiBoost - Multi-purpose boosting package
 *
 *    Copyright (C) 2010   AppStat group
 *                         Laboratoire de l'Accelerateur Lineaire
 *                         Universite Paris-Sud, 11, CNRS
 *
 *    This file is part of the MultiBoost library
 *
 *    This library is free software; you can redistribute it 
 *    and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation; either
 *    version 2.1 of the License, or (at your option) any later version.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
 *
 *    Contact: Balazs Kegl (balazs.kegl@gmail.com)
 *             Norman Casagrande (nova77@gmail.com)
 *             Robert Busa-Fekete (busarobi@gmail.com)
 *
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
 *
 */


#include <limits>

#include <math.h>

#include "OutputInfo.h"
#include "NameMap.h"
#include "WeakLearners/BaseLearner.h"
#include "Others/Example.h"
#include "Utils/Utils.h"

namespace MultiBoost {
	
	// -------------------------------------------------------------------------
	
	OutputInfo::OutputInfo(const string& outputInfoFile)
	{
		// open the stream
		_outStream.open(outputInfoFile.c_str());
		
		// is it really open?
		if ( !_outStream.is_open() )
		{
			cerr << "ERROR: cannot open the output steam (<" 
			<< outputInfoFile << ">) for the step-by-step info!" << endl;
			exit(1);
		}
	}
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputHeader()
	{ 
		// column names
		//_outStream << "t\tErrTrn\tErrTst\tMinMTrn\tErrMTrn\tEdge\tMinMTst\tErrMTst\tMAETrn\tMSETrn\tMAETst\tMSETst\tTime\n";
		//_outStream << "t\tErrTrn\tErrTst\tTime\n";
		_outStream << "t\tErrTrn\tErrTst\tErrWTrn\tErrWTst\tTime\n";
		// Intial values for 0th iteration. Basically it's because gnuplot fails
		// if there isn't any number in a column, which happens with "NA". On the other hand, 
		// NA is for having exactly the same number and order of columns all the time. 
		//    _outStream << "0\t1\t1\t-1\t1\tNA\t1\t1\n";
	}
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputIteration(int t)
	{ 
		_outStream << (t+1); // just output t
	}
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputCurrentTime( void )
	{ 
		time_t seconds;
		seconds = time (NULL);
		
		_outStream << '\t' << (seconds - _beginingTime); // just output current time in seconds
	}
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::initialize(InputData* pData)
	{ 
		_beginingTime = time( NULL );
		
		int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		table& g = _gTableMap[pData];
		g.resize(numExamples);
		
		for ( int i = 0; i < numExamples; ++i )
		{
			//if (pData->isSparseLabel())
			//   numClasses = pData->getNumNonzeroLabels(i);
			g[i].resize(numClasses);
			for (int l = 0; l < numClasses; ++l)
				g[i][l] = 0;
		}
		
		table& margins = _margins[pData];
		margins.resize(numExamples);
		
		for ( int i = 0; i < numExamples; ++i )
		{
			//if (pData->isSparseLabel())
			//   numClasses = pData->getNumNonzeroLabels(i);
			margins[i].resize(numClasses);
			for (int l = 0; l < numClasses; ++l)
				margins[i][l] = 0;
		}
		
		_alphaSums[pData] = 0;
	}
	
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputError(InputData* pData, BaseLearner* pWeakHypothesis)
	{
		int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		table& g = _gTableMap[pData];
		vector<Label>::const_iterator lIt;
		
		// Building the strong learner (discriminant function)
		for (int i = 0; i < numExamples; ++i)
		{
			const vector<Label>& labels = pData->getLabels(i);
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				g[i][lIt->idx] += pWeakHypothesis->getAlpha() * // alpha
				pWeakHypothesis->classify( pData, i, lIt->idx ); 
			}
		}
		
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {
		//      g[i][l] += pWeakHypothesis->getAlpha() * // alpha
		//                 pWeakHypothesis->classify( pData, i, l ); // h_l(x)
		//   }
		//}
		
		int numErrors = 0;   
		
		for (int i = 0; i < numExamples; ++i)
		{
			const vector<Label>& labels = pData->getLabels(i);
			
			// the vote of the winning negative class
			float maxNegClass = -numeric_limits<float>::max();
			// the vote of the winning positive class
			float minPosClass = numeric_limits<float>::max();
			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				// get the negative winner class
				if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
					maxNegClass = g[i][lIt->idx];
				
				// get the positive winner class
				if ( lIt->y > 0 && g[i][lIt->idx] < minPosClass )
					minPosClass = g[i][lIt->idx];
			}
			
			// if the vote for the worst positive label is lower than the
			// vote for the highest negative label -> error
			if (minPosClass <= maxNegClass)
				++numErrors;
		}
		
		//// Single label: error is when the argmax of the discriminant 
		//// function is not the correct class
		//if (pData->isSingleLabel())
		//{
		//   // Compute the "traditional" training error
		//   for (int i = 0; i < numExamples; ++i)
		//   {
		//      // If there is a tie, we count it as error
		//      bool equality = true;
		
		//      // the class with the highest vote
		//      int maxClassIdx = -1;
		
		//      // the vote of the winning class
		//      float maxClass = -numeric_limits<float>::max();
		
		//      for (int l = 0; l < numClasses; ++l)
		//      {     
		//         // get the winner class
		//         if (g[i][l] > maxClass + 1E-10)
		//         {
		//            maxClass = g[i][l];
		//            maxClassIdx = l;
		//            equality = false;
		//         }
		//         else if (g[i][l] > maxClass - 1E-10)
		//            equality = true;
		//      }
		
		//      // if the winner class is not the actual class, then it is
		//      // an error
		//      if (equality || maxClassIdx != pData->getClass(i))
		//         ++numErrors;
		//   }
		//}
		//// Multi label: correct classification is when the all positive 
		//// labels get higher votes than all negative labels
		//else 
		//{
		
		//   for (int i = 0; i < numExamples; ++i)
		//   {
		//      const int * labels = pData->getLabels(i);
		
		//      // the vote of the winning negative class
		//      float maxNegClass = -numeric_limits<float>::max();
		//      // the vote of the winning positive class
		//      float minPosClass = numeric_limits<float>::max();
		
		//      if (pData->isSparseLabel())
		//         numClasses = pData->getNumNonzeroLabels(i);
		
		//      for (int l = 0; l < numClasses; ++l)
		//      {     
		//         // get the negative winner class
		//         if (labels[l] < 0 && g[i][l] > maxNegClass)
		//            maxNegClass = g[i][l];
		
		//         // get the positive winner class
		//         if (labels[l] > 0 && g[i][l] < minPosClass)
		//            minPosClass = g[i][l];
		//      }
		
		//      // if the vote for the worst positive label is lower than the
		//      // vote for the highest negative label -> error
		//      if (minPosClass <= maxNegClass)
		//         ++numErrors;
		//   }
		//}
		
		// The error is normalized by the number of points
		_outStream << '\t' << (float)(numErrors)/(float)(numExamples);
		
	}
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputWeightedError(InputData* pData)
	{
		int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		table& g = _gTableMap[pData];
		vector<Label>::const_iterator lIt;
		
		float w = 0.0;
		float sumWeight = 0;
		int numErrors = 0;   
		
		for (int i = 0; i < numExamples; ++i)
		{
			const vector<Label>& labels = pData->getLabels(i);
			
			// the vote of the winning negative class
			float maxNegClass = -numeric_limits<float>::max();
			// the vote of the winning positive class
			float minPosClass = numeric_limits<float>::max();
			
			float posWeight = numeric_limits<float>::max();
			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				// get the negative winner class
				if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
					maxNegClass = g[i][lIt->idx];
				
				// get the positive winner class
				if ( lIt->y > 0 && g[i][lIt->idx] < minPosClass ) {
					minPosClass = g[i][lIt->idx];
					//#ifdef NOTIWEIGHT					
					//posWeight = lIt->weight;
					//#else					
					posWeight = lIt->initialWeight;
					//#endif
				}
			}
			
			// if the vote for the worst positive label is lower than the
			// vote for the highest negative label -> error
			if (minPosClass <= maxNegClass) {
				++numErrors; // just indicating that there is missclassification in this case				
			} else {
				w += posWeight;
			}
			sumWeight += posWeight;
		}
		// The error is normalized by the sum of positive weights
		_outStream << '\t' << 1-(w/sumWeight);
		
	}
	
	
	
	// -------------------------------------------------------------------------
	// http://www.kddcup-orange.com/evaluation.php
	void OutputInfo::outputBalancedError(InputData* pData)
	{
		int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		vector< int > tp( numClasses );   
		fill( tp.begin(), tp.end(), 0 );
		
		vector< int > tn( numClasses );   
		fill( tn.begin(), tn.end(), 0 );
		
		vector< float > bacPerClass( numClasses );   
		fill( bacPerClass.begin(), bacPerClass.end(), 0.0 );
		
		table& g = _gTableMap[pData];
		vector<Label>::const_iterator lIt;
		
		// Building the strong learner (discriminant function)
		//for (int i = 0; i < numExamples; ++i)
		//{
		//	const vector<Label>& labels = pData->getLabels(i);
		
		//	for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
		//	{
		//		g[i][lIt->idx] += pWeakHypothesis->getAlpha() * // alpha
		//			pWeakHypothesis->classify( pData, i, lIt->idx ); 
		//	}
		//}
		
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {
		//      g[i][l] += pWeakHypothesis->getAlpha() * // alpha
		//                 pWeakHypothesis->classify( pData, i, l ); // h_l(x)
		//   }
		//}
		
		
		for (int i = 0; i < numExamples; ++i)
		{
			const vector<Label>& labels = pData->getLabels(i);
			
			// the vote of the winning negative class
			float maxNegClass = -numeric_limits<float>::max();
			// the vote of the winning positive class
			float minPosClass = numeric_limits<float>::max();
			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				// get the negative winner class
				if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
					maxNegClass = g[i][lIt->idx];
				
				// get the positive winner class
				if ( lIt->y > 0 && g[i][lIt->idx] < minPosClass )
					minPosClass = g[i][lIt->idx];
				
			}
			
			// if the vote for the worst positive label is higher than the
			// vote for the highest negative label -> good label
			if (minPosClass > maxNegClass){
				for ( lIt = labels.begin(); lIt != labels.end(); ++lIt ) {
					if ( lIt->y > 0  ) {
						tp[ lIt->idx]++;
					} else { 
						tn[ lIt->idx]++;
					}
				}
			}
			
		}
		
		float bACC = 0.0;
		
		for( int i = 0; i < numClasses; i++ ) {
			float specificity = (((float)tp[i]) / ((float) pData->getNumExamplesPerClass( i ) ));
			float sensitivity = ( ((float)tn[i]) / ((float) ( numExamples - pData->getNumExamplesPerClass( i ))) );
			//_outStream << '\t' << numExamples - pData->getNumExamplesPerClass( i );
			//_outStream << '\t' << specificity << '\t' << sensitivity;
			bacPerClass[ i ] = 0.5 * ( specificity + sensitivity );
			bACC += bacPerClass[i];
		}
		
		bACC /= (float) numClasses;
		
		_outStream << '\t' << bACC;
		
		for( int i = 0; i < numClasses; i++ ) {
			_outStream << '\t' << bacPerClass[i];
		}
		
		
	}
	
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputMAE(InputData* pData)
	{
		const int numExamples = pData->getNumExamples();
		
		table& g = _gTableMap[pData];
		vector<Label>::const_iterator lIt,maxlIt,truelIt;
		float maxDiscriminant,mae = 0.0,mse = 0.0,tmpVal;
		char maxLabel;
		
		// Get label values: they must be convertible to float
		vector<float> labelValues;
		NameMap classMap = pData->getClassMap();
		for (int l = 0;l < classMap.getNumNames(); ++l)
			labelValues.push_back(atof(classMap.getNameFromIdx(l).c_str()));
		
		// Building the strong learner (discriminant function)
		for (int i = 0; i < numExamples; ++i){
			const vector<Label>& labels = pData->getLabels(i);
			maxDiscriminant = -numeric_limits<float>::max();
			maxLabel = -100;
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt ) {
				if ( g[i][lIt->idx] > maxDiscriminant ) {
					maxDiscriminant = g[i][lIt->idx];
					maxlIt = lIt;
				}
				if ( lIt->y > maxLabel ) {
					maxLabel = lIt->y;
					truelIt = lIt;
				}	 
			}
			tmpVal = labelValues[truelIt->idx] - labelValues[maxlIt->idx];
			mae += fabs(tmpVal);				      
			mse += tmpVal * tmpVal;				      
		}
		
		_outStream << '\t' << mae/(float)(numExamples) << '\t' << sqrt(mse/(float)(numExamples));
		
	}
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputMargins(InputData* pData, BaseLearner* pWeakHypothesis )
	{
		int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		//    ITT FOLYTATNI: single/dense/sparse
		
		table& margins = _margins[pData];
		
		float minMargin = numeric_limits<float>::max();
		float belowZeroMargin = 0;                        
		
		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				float hy =  pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
				lIt->y; // y
				
				// compute the margin
				margins[i][lIt->idx] += pWeakHypothesis->getAlpha() * hy;
				
				// gets the margin below zero
#ifdef NOTIWEIGHT
				if ( margins[i][lIt->idx] < 0 )
					belowZeroMargin += lIt->weight;
#else
				if ( margins[i][lIt->idx] < 0 )
					belowZeroMargin += lIt->initialWeight;
#endif
				
				// get the minimum margin among classes and examples
				if (margins[i][lIt->idx] < minMargin)
					minMargin = margins[i][lIt->idx];
			}
		}
		
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {
		//      // hy = +1 if the classification it is correct, -1 otherwise
		//      float hy = pWeakHypothesis->classify(pData, i, l) * // h_l(x_i)
		//                  pData->getLabel(i, l); // y_{i,l}
		
		//      // compute the margin
		//      margins[i][l] += pWeakHypothesis->getAlpha() * hy;
		
		//      // gets the margin below zero
		//      if ( margins[i][l] < 0 )
		//      {
		//         if (l == pData->getClass(i))
		//            belowZeroMargin += ( 1.0 / static_cast<float>(2*numExamples) );
		//         else
		//            belowZeroMargin += ( 1.0 / static_cast<float>(2*numExamples * (numClasses - 1)) );
		//      }        
		
		//      // get the minimum margin among classes and examples
		//      if (margins[i][l] < minMargin)
		//         minMargin = margins[i][l];
		//   }
		//}
		
		// compute the sums of the alphas for normalization
		_alphaSums[pData] += pWeakHypothesis->getAlpha();
		
		_outStream << '\t' << minMargin/_alphaSums[pData] << "\t" // minimum margin
		<< belowZeroMargin; // margins that are below zero
	}
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputEdge(InputData* pData, BaseLearner* pWeakHypothesis)
	{
		const int numExamples = pData->getNumExamples();
		
		float gamma = 0; // the edge
		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				float hy = pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
				lIt->y;
				gamma += lIt->weight * hy;
			}
		}
		
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {      
		//      // hy = +1 if the classification it is correct, -1 otherwise
		//      float hy = pWeakHypothesis->classify(pData, i, l) * // h_l(x_i)
		//                  pData->getLabel(i, l); // y_i
		
		//      float w = pData->getWeight(i, l);
		
		//      gamma += w * hy;
		//   }
		//}
		
		_outStream << '\t' << gamma; // edge
		
	}
	
	// -------------------------------------------------------------------------
	
	void OutputInfo::outputROC(InputData* pData)
	{
		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		vector< int > fp( numClasses );   
		fill( fp.begin(), fp.end(), 0 );
		
		table& g = _gTableMap[pData];
		vector<Label>::const_iterator lIt;
		
		//// Building the strong learner (discriminant function)
		//for (int i = 0; i < numExamples; ++i)
		//{
		//	const vector<Label>& labels = pData->getLabels(i);
		//
		//	for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
		//	{
		//		g[i][lIt->idx] += pWeakHypothesis->getAlpha() * // alpha
		//			pWeakHypothesis->classify( pData, i, lIt->idx ); 
		//	}
		//}
		
		//vector< double > scores( numExamples );
		//vector< int > labels( numExamples );
		
		vector< pair< int, double > > data( numExamples );
		
		vector< double > ROCscores( numClasses );
		fill( ROCscores.begin(), ROCscores.end(), 0.0 );
		double ROCsum = 0.0;
		
		for( int i=0; i < numClasses; i++ ) {
			if ( 0 < pData->getNumExamplesPerClass( i ) ) {
				
				//fill( labels.begin(), labels.end(), 0 );
				double mn = numeric_limits< double >::max();
				double mx = numeric_limits< double >::min();
				
				
				
				for( int j = 0; j < numExamples; j++ ) {
					data[j].second = g[j][i];
					
					if ( mn > data[j].second ) mn = data[j].second;
					if ( mx < data[j].second ) mx = data[j].second;					
					
					if ( pData->hasPositiveLabel( j, i ) ) data[j].first = 1;
					else data[j].first = 0;
				}
				
				mx -= mn;
				if ( mx > numeric_limits<double>::epsilon() ) {
					for( int j = 0; j < numExamples; j++ ) {
						data[j].second -= mn;
						data[j].second /= mx; 
					}
				}
				
				ROCscores[i] = nor_utils::getROC( data );
			} else {
				ROCscores[i] = 0.0;
			}
			
			ROCsum += ROCscores[i];
		}
		ROCsum /= (double) numClasses;
		
		_outStream << '\t' << ROCsum; // mean of AUC
		for( int i=0; i < numClasses; i++ ) {
			_outStream << '\t' << ROCscores[i];
		}
		
	}
	
	// -------------------------------------------------------------------------	
	void OutputInfo::outTPRFPR( InputData* pData )
	{
		int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		table& g = _gTableMap[pData];
		
		vector<int> origLabels(numExamples);
		vector<int> forecastedLabels(numExamples);
		
		for (int i = 0; i < numExamples; ++i)
		{
			const vector<Label>& labels = pData->getLabels(i);
			
			vector<Label>::const_iterator lIt;
			
			// the vote of the winning negative class
			float maxClass = -numeric_limits<float>::max();
			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				// get the negative winner class
				if ( g[i][lIt->idx] > maxClass )
				{
					forecastedLabels[i]=lIt->idx;
					maxClass = g[i][lIt->idx];
				}
				
				// get the positive winner class
				if ( lIt->y > 0 )
				{
					origLabels[i]=lIt->idx;
				}
			}						
		}
		
		vector<double> TPR(numClasses);
		vector<double> FPR(numClasses);
		
		vector<int> TP(numClasses);
		vector<int> FP(numClasses);
		
		vector<int> classDistr(numClasses);
		
		fill( classDistr.begin(), classDistr.end(), 0 );
		fill( TP.begin(), TP.end(), 0 );
		fill( FP.begin(), FP.end(), 0 );
		
		for (int i = 0; i < numExamples; ++i)
		{
			classDistr[origLabels[i]]++;
			
			if (origLabels[i]==forecastedLabels[i]) // True positive
			{
				TP[origLabels[i]]++;
			} else {
				FP[forecastedLabels[i]]++;
			}			
		}		
		
		double avgTPR = 0.0, avgFPR = 0.0;
		// print
		for( int l=0; l<numClasses; ++l ) {
			TPR[l] = TP[l]/((double)classDistr[l]);
			FPR[l] = FP[l]/((double)(numExamples-classDistr[l]));
			
			avgTPR += TPR[l];
			avgFPR += FPR[l];
		}
		
		avgTPR /= numClasses;
		avgFPR /= numClasses;
		
		_outStream << '\t' << avgTPR; // mean of TPR
		for( int i=0; i < numClasses; i++ ) {
			_outStream << '\t' << TPR[i];
		}
		
		_outStream << '\t' << avgFPR; // mean of FPR
		for( int i=0; i < numClasses; i++ ) {
			_outStream << '\t' << FPR[i];
		}				
	}
    // -------------------------------------------------------------------------	
	
} // end of namespace MultiBoost
