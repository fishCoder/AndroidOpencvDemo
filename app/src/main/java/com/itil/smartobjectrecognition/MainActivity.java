package com.itil.smartobjectrecognition;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener, CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    /*
     * Main MenuItems die nach dem Klicken auf die Setting auf dem Bildschirm erscheinen
     *
     */
    private MenuItem mItemDetector;        // Detector MenuItem
    private MenuItem mItemDescriptor;    // DescdriptorExtractor MenuItem
    private MenuItem mItemMatcher;        // DescriptorMatcher MenuItem
    private MenuItem mItemFunction;        // Function MenuItem

	/*
	 * Mat die f¸r die Objekterkennung genutzt werden. Sie sind in RGB und Grauwert vorhanden.
	 * imageSceneRgba wird von der Kamera geliefert und imageObjectRgba wird mittels OnTouch-Funktion des Handys
	 * ausgew‰hlt.
	 */

    private Mat imageSceneRgba;            // Kamerabild in Rgba format
    private Mat imageSceneGray;            // Kamerabild in Grauwert

    private Mat imageObjectGray;        // Objektbild in Grauwert

    int mFunctionMode = 0;

    private Rect touchedRect;            // Auswahlrecteck mittels Ber¸hren auf dem Bildschirm
    private Point touchPoint;            // Ber¸hrpunkt auf dem Bildschirm

    private boolean mIsObjectSelected = false;    // Wenn auf das Bildschirm getippt wird, wird der darunter liegende Bild im Rechteck als Objekt ausgew‰hlt. Is ein Objektbild ausgew‰hlt?
    private boolean mDrawSelectedRect = false;    // soll der Ausgew‰hlte Rechteck kurz in einem Frame erscheinen

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.d(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization


                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        super();
    }

    static {
        OpenCVLoader.initDebug();
        System.loadLibrary("native-lib");
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);



        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.smartobjectRecognitionActivity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mLoaderCallback.onManagerConnected(0);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_10, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        imageSceneRgba = new Mat(height, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        imageSceneRgba.release();
        if (imageSceneGray != null) {
            imageSceneGray.release();
            imageSceneGray = null;
        }
        if (imageObjectGray != null) {
            imageObjectGray.release();
            imageObjectGray = null;
        }
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // input frame has RGBA format
        imageSceneRgba = inputFrame.rgba();
        imageSceneGray = inputFrame.gray();

        try {
            if (mDrawSelectedRect) {
                Core.rectangle(imageSceneRgba,
                        new Point(touchedRect.x, touchedRect.y),
                        new Point(touchedRect.x + touchedRect.width, touchedRect.y + touchedRect.height),
                        new Scalar(0, 255, 255), 1);
                mDrawSelectedRect = false;
            }

            switch (mFunctionMode) {
                case 1:
                    if (mItemDetector != null) {
                        FindFeatures(imageSceneGray.getNativeObjAddr(), imageSceneRgba.getNativeObjAddr(), mItemDetector.getTitle().toString());
                    }

                    break;
                case 2:
                    if (mIsObjectSelected) {
                        if (mItemDetector != null && mItemDescriptor != null && mItemMatcher != null) {
                            javaFindObject(mItemDetector.getTitle().toString(), mItemDescriptor.getTitle().toString(), mItemMatcher.getTitle().toString());
                        }
                    }
                    break;
                case 3:
                    if (mIsObjectSelected) {
                        if (mItemDetector != null && mItemDescriptor != null && mItemMatcher != null) {

                            FindObject(imageSceneGray.getNativeObjAddr(),
                                    imageSceneRgba.getNativeObjAddr(),
                                    imageObjectGray.getNativeObjAddr(),
                                    mItemDetector.getTitle().toString(),
                                    mItemDescriptor.getTitle().toString(),
                                    mItemMatcher.getTitle().toString());

                        }
                    }

                default:
                    break;
            }
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }

        return imageSceneRgba;

    }


    public void javaFindObject(String detectorName, String extractorName, String matcherName) {

        try {

            FeatureDetector detector = FeatureDetector.create(getFeatureDetectorType(detectorName));
            DescriptorExtractor extractor = DescriptorExtractor.create(getDescriptorExtractorType(extractorName));
            DescriptorMatcher matcher = DescriptorMatcher.create(getDescriptorMatcherType(matcherName));

            // -- Step 1: Detect the keypoints using Detector
            MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
            MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();
            detector.detect(imageObjectGray, keypoints_object);
            detector.detect(imageSceneGray, keypoints_scene);

            // -- Step 2: Calculate descriptors (feature vectors)
            Mat descriptors_object = new Mat();
            Mat descriptors_scene = new Mat();
            extractor.compute(imageObjectGray, keypoints_object, descriptors_object);
            extractor.compute(imageSceneGray, keypoints_scene, descriptors_scene);

            // -- Step 3: Matching descriptor vectors using matcher
            MatOfDMatch matches = new MatOfDMatch();
            matcher.match(descriptors_object, descriptors_scene, matches);

            List<DMatch> matchesList = matches.toList();
            double max_dist = 0;
            double min_dist = 100;
            // -- Quick calculation of max and min distances between keypoints
            for (int i = 0; i < descriptors_object.rows(); i++) {
                double dist = matchesList.get(i).distance;
                if (dist < min_dist) {
                    min_dist = dist;
                }
                if (dist > max_dist) {
                    max_dist = dist;
                }
            }

            // -- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
            Vector<DMatch> good_matches = new Vector<DMatch>();
            for (int i = 0; i < descriptors_object.rows(); i++) {
                if (matchesList.get(i).distance < 3 * min_dist) {
                    good_matches.add(matchesList.get(i));
                }
            }

            List<Point> objListGoodMatches = new ArrayList<Point>();
            List<Point> sceneListGoodMatches = new ArrayList<Point>();

            List<KeyPoint> keypoints_objectList = keypoints_object.toList();
            List<KeyPoint> keypoints_sceneList = keypoints_scene.toList();

            for (int i = 0; i < good_matches.size(); i++) {
                // -- Get the keypoints from the good matches
                objListGoodMatches.add(keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
                sceneListGoodMatches.add(keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
                Core.circle(imageSceneRgba, new Point(sceneListGoodMatches.get(i).x, sceneListGoodMatches.get(i).y), 3, new Scalar(255, 0, 0, 255));

            }
            String text = "Good Matches Count: " + good_matches.size();
            Core.putText(imageSceneRgba, text, new Point(0, 60), Core.FONT_HERSHEY_COMPLEX_SMALL, 1, new Scalar(0, 0, 255, 255));


            MatOfPoint2f objListGoodMatchesMat = new MatOfPoint2f();
            objListGoodMatchesMat.fromList(objListGoodMatches);
            MatOfPoint2f sceneListGoodMatchesMat = new MatOfPoint2f();
            sceneListGoodMatchesMat.fromList(sceneListGoodMatches);

            // findHomography needs 4 corresponding points
            if (good_matches.size() > 3) {


                Mat H = Calib3d.findHomography(objListGoodMatchesMat, sceneListGoodMatchesMat, Calib3d.RANSAC, 5 /* RansacTreshold */);

                Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
                Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

                obj_corners.put(0, 0, new double[]{0, 0});
                obj_corners.put(1, 0, new double[]{imageObjectGray.cols(), 0});
                obj_corners.put(2, 0, new double[]{imageObjectGray.cols(), imageObjectGray.rows()});
                obj_corners.put(3, 0, new double[]{0, imageObjectGray.rows()});

                Core.perspectiveTransform(obj_corners, scene_corners, H);

                Core.line(imageSceneRgba, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 2);
                Core.line(imageSceneRgba, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), new Scalar(0, 255, 0), 2);
                Core.line(imageSceneRgba, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), new Scalar(0, 255, 0), 2);
                Core.line(imageSceneRgba, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), new Scalar(0, 255, 0), 2);

            }
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }


    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        super.onPrepareOptionsMenu(menu);
        Log.i(TAG, "called onPrepareOptionsMenu");

        menu.clear();
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_main, menu);

        if (mItemDetector != null) {
            menu.findItem(mItemDetector.getItemId()).setChecked(true);
        }
        if (mItemDescriptor != null) {
            menu.findItem(mItemDescriptor.getItemId()).setChecked(true);
        }
        if (mItemMatcher != null) {
            menu.findItem(mItemMatcher.getItemId()).setChecked(true);
        }

        if (mItemFunction != null) {
            menu.findItem(mItemFunction.getItemId()).setChecked(true);
        }



        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        item.setChecked(true);
        int id = item.getItemId();

        if (id == R.id.detector ||
                id == R.id.descriptorExtractor ||
                id == R.id.matcher) {
            return true;
        }

        if (item.getGroupId() == R.id.group_detector) {
            mItemDetector = item;
            return true;
        }

        if (item.getGroupId() == R.id.group_Descriptor) {

            mItemDescriptor = item;
            return true;
        }

        if (item.getGroupId() == R.id.group_Matcher) {
            mItemMatcher = item;
            return true;
        }

        if (id == R.id.findFeature) {
            mItemFunction = item;
            mFunctionMode = 1;
            return true;
        }
        if (id == R.id.javaFindObject) {
            mItemFunction = item;
            mFunctionMode = 2;
            return true;
        }

        if (id == R.id.nativeFindObject) {
            mItemFunction = item;
            mFunctionMode = 3;
            return true;
        }
        return true;
    }

    public boolean onTouch(View v, MotionEvent event) {
        int cols = imageSceneRgba.cols();
        int rows = imageSceneRgba.rows();

        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

        int x = (int) event.getX() - xOffset;
        int y = (int) event.getY() - yOffset;

        Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

        touchPoint = new Point(x, y);

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        touchedRect = new Rect();

        touchedRect.x = (x > 128) ? x - 128 : 0;
        touchedRect.y = (y > 128) ? y - 128 : 0;

        touchedRect.width = (x + 128 < cols) ? x + 128 - touchedRect.x : cols - touchedRect.x;
        touchedRect.height = (y + 128 < rows) ? y + 128 - touchedRect.y : rows - touchedRect.y;

        mDrawSelectedRect = true;
        Mat imageObjectRgba = imageSceneRgba.submat(touchedRect);
        imageObjectGray = new Mat(imageObjectRgba.height(), imageObjectRgba.width(), CvType.CV_8UC1);
        Imgproc.cvtColor(imageObjectRgba, imageObjectGray, Imgproc.COLOR_RGB2GRAY);

        mIsObjectSelected = true;

        return false; // don't need subsequent touch events
    }

    /**
     * Returns detectorType to a given detectorName;
     *
     * @param detectorName See implementation
     * @return detectorType
     */
    public int getFeatureDetectorType(String detectorName) {

        if (detectorName.equals("SIFT")) {
            return FeatureDetector.SIFT;
        } else if (detectorName.equals("SURF")) {
            return FeatureDetector.SURF;
        } else if (detectorName.equals("ORB")) {
            return FeatureDetector.ORB;
        } else if (detectorName.equals("FAST")) {
            return FeatureDetector.FAST;
        } else if (detectorName.equals("STAR")) {
            return FeatureDetector.STAR;
        } else if (detectorName.equals("MSER")) {
            return FeatureDetector.MSER;
        } else if (detectorName.equals("GFTT")) {
            return FeatureDetector.GFTT;
        } else if (detectorName.equals("Dense")) {
            return FeatureDetector.DENSE;
        } else if (detectorName.equals("BRISK")) {
            return FeatureDetector.BRISK;
        } else if (detectorName.equals("HARRIS")) {
            return FeatureDetector.HARRIS;
        } else if (detectorName.equals("SimpleBlob")) {
            return FeatureDetector.SIMPLEBLOB;
        } else if (detectorName.equals("GridSIFT")) {
            return FeatureDetector.GRID_SIFT;
        } else if (detectorName.equals("GridSURF")) {
            return FeatureDetector.GRID_SURF;
        } else if (detectorName.equals("GridORB")) {
            return FeatureDetector.GRID_ORB;
        } else if (detectorName.equals("GridFAST")) {
            return FeatureDetector.GRID_FAST;
        } else if (detectorName.equals("GridSTAR")) {
            return FeatureDetector.GRID_STAR;
        } else if (detectorName.equals("GridMSER")) {
            return FeatureDetector.GRID_MSER;
        } else if (detectorName.equals("GridGFTT")) {
            return FeatureDetector.GRID_GFTT;
        } else if (detectorName.equals("GridDense")) {
            return FeatureDetector.GRID_DENSE;
        } else if (detectorName.equals("GridBRISK")) {
            return FeatureDetector.GRID_BRISK;
        } else if (detectorName.equals("GridHARRIS")) {
            return FeatureDetector.GRID_HARRIS;
        } else if (detectorName.equals("GridSimpleBlob")) {
            return FeatureDetector.GRID_SIMPLEBLOB;
        } else if (detectorName.equals("PyramidSIFT")) {
            return FeatureDetector.PYRAMID_SIFT;
        } else if (detectorName.equals("PyramidSURF")) {
            return FeatureDetector.PYRAMID_SURF;
        } else if (detectorName.equals("PyramidORB")) {
            return FeatureDetector.PYRAMID_ORB;
        } else if (detectorName.equals("PyramidFAST")) {
            return FeatureDetector.PYRAMID_FAST;
        } else if (detectorName.equals("PyramidSTAR")) {
            return FeatureDetector.PYRAMID_STAR;
        } else if (detectorName.equals("PyrmaidMSER")) {
            return FeatureDetector.PYRAMID_MSER;
        } else if (detectorName.equals("PyramidGFTT")) {
            return FeatureDetector.PYRAMID_GFTT;
        } else if (detectorName.equals("PyramidDense")) {
            return FeatureDetector.PYRAMID_DENSE;
        } else if (detectorName.equals("PyrmaidBRISK")) {
            return FeatureDetector.PYRAMID_BRISK;
        } else if (detectorName.equals("PyramidHARRIS")) {
            return FeatureDetector.PYRAMID_HARRIS;
        } else if (detectorName.equals("PyramidSimpleBlob")) {
            return FeatureDetector.PYRAMID_SIMPLEBLOB;
        } else if (detectorName.equals("DynamicSIFT")) {
            return FeatureDetector.DYNAMIC_SIFT;
        } else if (detectorName.equals("DynamicSURF")) {
            return FeatureDetector.DYNAMIC_SURF;
        } else if (detectorName.equals("DynamicORB")) {
            return FeatureDetector.DYNAMIC_ORB;
        } else if (detectorName.equals("DynamicFAST")) {
            return FeatureDetector.DYNAMIC_FAST;
        } else if (detectorName.equals("DynamicSTAR")) {
            return FeatureDetector.DYNAMIC_STAR;
        } else if (detectorName.equals("DynamicMSER")) {
            return FeatureDetector.DYNAMIC_MSER;
        } else if (detectorName.equals("DynamicGFTT")) {
            return FeatureDetector.DYNAMIC_GFTT;
        } else if (detectorName.equals("DynamicDense")) {
            return FeatureDetector.DYNAMIC_DENSE;
        } else if (detectorName.equals("DynamicBRISK")) {
            return FeatureDetector.DYNAMIC_BRISK;
        } else if (detectorName.equals("DynamicHARRIS")) {
            return FeatureDetector.DYNAMIC_HARRIS;
        } else if (detectorName.equals("DynamicSimpleBlob")) {
            return FeatureDetector.DYNAMIC_SIMPLEBLOB;
        } else {
            return 0;
        }
    }

    public int getDescriptorExtractorType(String extractorName) {
        if (extractorName.equals("SURF")) {
            return DescriptorExtractor.SURF;
        } else if (extractorName.equals("SIFT")) {
            return DescriptorExtractor.SIFT;
        } else if (extractorName.equals("ORB")) {
            return DescriptorExtractor.ORB;
        } else if (extractorName.equals("BRIEF")) {
            return DescriptorExtractor.BRIEF;
        } else if (extractorName.equals("BRISK")) {
            return DescriptorExtractor.BRISK;
        } else if (extractorName.equals("FREAK")) {
            return DescriptorExtractor.FREAK;
        } else if (extractorName.equals("OpponentSIFT")) {
            return DescriptorExtractor.OPPONENT_SIFT;
        } else if (extractorName.equals("OpponentSURF")) {
            return DescriptorExtractor.OPPONENT_SURF;
        } else if (extractorName.equals("OpponentORB")) {
            return DescriptorExtractor.OPPONENT_ORB;
        } else if (extractorName.equals("OpponentBRIEF")) {
            return DescriptorExtractor.OPPONENT_BRIEF;
        } else if (extractorName.equals("OpponentBRISK")) {
            return DescriptorExtractor.OPPONENT_BRISK;
        } else if (extractorName.equals("OpponentFREAK")) {
            return DescriptorExtractor.OPPONENT_FREAK;
        } else {
            return 0;
        }
    }

    public int getDescriptorMatcherType(String matcherName) {

        if (matcherName.equals("BruteForce")) {
            return DescriptorMatcher.BRUTEFORCE;
        } else if (matcherName.equals("BruteForce-L1")) {
            return DescriptorMatcher.BRUTEFORCE_L1;
        } else if (matcherName.equals("BruteForce-Hamming")) {
            return DescriptorMatcher.BRUTEFORCE_HAMMING;
        } else if (matcherName.equals("FlannBased")) {
            return DescriptorMatcher.FLANNBASED;
        } else if (matcherName.equals("BruteForche-Hamming LUT")) {
            return DescriptorMatcher.BRUTEFORCE_HAMMINGLUT;
        } else if (matcherName.equals("BruteForce SL2")) {
            return DescriptorMatcher.BRUTEFORCE_SL2;
        } else {
            return 0;
        }

    }

    public native void FindFeatures(long matAddrGr, long matAddrRgba, String detectorName);

    public native void FindObject(long matAddrGr, long matAddrRgba, long addrObjGray, String detectorName, String descriptorName, String matcherName);
}