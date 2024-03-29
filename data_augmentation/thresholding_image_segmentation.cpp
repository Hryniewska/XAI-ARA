#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 45;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );

/** @function main */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  src = imread( argv[1], 1 );
  resize(src, src, Size(src.cols*3, src.rows*3));

  /// Convert image to gray and blur it
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );
  

  /// Create Window
  const char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 0, 0 );

  waitKey(0);
  return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  /// Find contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Find the rotated rectangles and ellipses for each contour
  // vector<RotatedRect> minRect( contours.size() );
  // vector<RotatedRect> minEllipse( contours.size() );

  // for( int i = 0; i < contours.size(); i++ )
  //    { minRect[i] = minAreaRect( Mat(contours[i]) );
  //      if( contours[i].size() > 5 )
  //        { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
  //    }


  /// Draw contours + rotated rects + ellipses
  // Mat drawing = src.clone();
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

  int largest_area=0;
  int largest_contour_index=0;
  for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
  {
    double a=contourArea( contours[i],false);  //  Find the area of contour
    if(a>largest_area){
        largest_area=a;
        largest_contour_index=i;                //Store the index of largest contour
    }
  }
  // cout << largest_area << " " << largest_contour_index << endl;
  contours[largest_contour_index].clear();
      

  Scalar color = Scalar(255,255,255,0);
  for( int i = 0; i< contours.size(); i++ )
     {
      //  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       // contour
       //drawContours(InputOutputArray image, InputArrayOfArrays contours, int contourIdx, const Scalar& color, int thickness=1, int lineType=8, InputArray hierarchy=noArray(), int maxLevel=INT_MAX, Point offset=Point() )
       drawContours( drawing, contours, i, color, CV_FILLED, 8, vector<Vec4i>(), 2, Point() );
       // ellipse
      //  ellipse( drawing, minEllipse[i], color, 2, 8 );
       // rotated rectangle
      //  Point2f rect_points[4]; minRect[i].points( rect_points );
      //  for( int j = 0; j < 4; j++ )
          // line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
     }

  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}