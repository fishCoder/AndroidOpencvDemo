#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;

extern "C" {

inline string intToString(int i)
{
    stringstream s;
    s << i;
    return s.str();
}

JNIEXPORT void JNICALL Java_com_itil_smartobjectrecognition_MainActivity_FindObject(JNIEnv* env, jobject, jlong addrGray, jlong addrRgba, jlong addrObjGray, jstring detectorName,  jstring descriptorName, jstring matcherName);

JNIEXPORT void JNICALL Java_com_itil_smartobjectrecognition_MainActivity_FindFeatures(JNIEnv* env, jobject, jlong addrGray, jlong addrRgba, jstring detectorName);

JNIEXPORT void JNICALL Java_com_itil_smartobjectrecognition_MainActivity_FindFeatures(JNIEnv* env, jobject, jlong addrGray, jlong addrRgba, jstring detectorName)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> v;

    const char * detector_name = (*env).GetStringUTFChars(detectorName , NULL ) ;
    Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);


    detector->detect(mGr, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(mRgb, Point(kp.pt.x, kp.pt.y), 3, Scalar(255,0,0,255));
    }

    string  text = "Features Count: " + intToString(v.size());
    putText(mRgb,text, Point(0,60),FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0,0,255,255), 1, 8);
}

JNIEXPORT void JNICALL Java_com_itil_smartobjectrecognition_MainActivity_FindObject(JNIEnv* env, jobject, jlong addrGray, jlong addrRgba, jlong addrObjGray, jstring detectorName,  jstring descriptorName, jstring matcherName){

    const char * detector_name = (*env).GetStringUTFChars(detectorName , NULL ) ;
    const char * descriptor_name = (*env).GetStringUTFChars(descriptorName , NULL ) ;
    const char * matcher_name = (*env).GetStringUTFChars(matcherName , NULL ) ;

    try{
        Mat& imageSceneGray  = *(Mat*)addrGray;
        Mat& imageSceneRgba = *(Mat*)addrRgba;
        Mat& imageObjectGray  = *(Mat*)addrObjGray;




        Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
        Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(descriptor_name);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_name);


        //-- Step 1: Detect the keypoints using Detector
        vector<KeyPoint> keypoints_object, keypoints_scene;
        detector->detect( imageObjectGray, keypoints_object );
        detector->detect( imageSceneGray, keypoints_scene );

        //-- Step 2: Calculate descriptors (feature vectors)
        Mat descriptors_object, descriptors_scene;
        extractor->compute( imageObjectGray, keypoints_object, descriptors_object );
        extractor->compute( imageSceneGray, keypoints_scene, descriptors_scene );

        //-- Step 3: Matching descriptor vectors using matcher
        vector< DMatch > matches;
        matcher->match( descriptors_object, descriptors_scene, matches );

        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_object.rows; i++ ){
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        vector< DMatch > good_matches;
        for( int i = 0; i < descriptors_object.rows; i++ ){
            if( matches[i].distance < 3*min_dist ){
                good_matches.push_back( matches[i]);
            }
        }

        vector<Point2f> obj;
        vector<Point2f> scene;
        for( unsigned int i = 0; i < good_matches.size(); i++ ){
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
            circle(imageSceneRgba, Point(scene[i].x, scene[i].y), 3, Scalar(255,0,0,255));
        }


        string  text = "Good matches count: " + intToString(good_matches.size());
        putText(imageSceneRgba,text, Point(0,60),FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0,0,255,255), 1, 8);

        // findHomography needs 4 corresponding points
        if(good_matches.size()>3){
            Mat H = findHomography( obj, scene, CV_RANSAC );

            //-- Get the corners from the image_1 ( the object to be "detected" )
            vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( imageObjectGray.cols, 0 );
            obj_corners[2] = cvPoint( imageObjectGray.cols, imageObjectGray.rows ); obj_corners[3] = cvPoint( 0, imageObjectGray.rows );
            vector<Point2f> scene_corners(4);


            perspectiveTransform( obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - img_camera )
            line( imageSceneRgba, scene_corners[0], scene_corners[1], Scalar( 0, 255, 0), 2 );
            line( imageSceneRgba, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 2 );
            line( imageSceneRgba, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 2 );
            line( imageSceneRgba, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 2 );

        }
    }catch(exception& e){
        e.what();
    }

}

}
