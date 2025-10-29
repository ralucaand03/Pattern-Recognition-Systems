#include "stdafx.h"
#include "stdio.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <ctime> 
#include <fstream>
#include <iomanip>  
using namespace std;
using namespace cv;

wchar_t* projectPath;
const int BLACK = 0;
const int WHITE = 255;
//------------------------------|| LAB 3 - Hough Transform ||------------------------------
struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};
void drawHoughLines(Mat& originalEdgeImg, const vector<peak>& peaks) {

	Mat colorImg;
	cvtColor(originalEdgeImg, colorImg, COLOR_GRAY2BGR);

	for (const auto& p : peaks) {
		double rho = p.ro;
		double thetaRad = (p.theta * CV_PI) / 180.0;

		double a = cos(thetaRad), b = sin(thetaRad);
		double x0 = a * rho, y0 = b * rho;

		Point pt1, pt2;

		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));

		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		line(colorImg, pt1, pt2, Scalar(200, 0, 200), 1);
	}

	imshow("Detected Lines", colorImg);
	//Mat enlarged;
	//resize(colorImg, enlarged, Size(), 3 , 3 );
	//imshow("Detected Lines", enlarged); 
}
void hough(Mat& img,int w,int k) {
	// 1.
	int width = img.cols;
	int height = img.rows;

	int D = (int)round(sqrt(width * width + height * height)); 
	Mat Hough = Mat::zeros(D + 1, 360, CV_32SC1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img.at<uchar>(y, x) == 255) {
				for (int theta = 0; theta < 360; theta++) {
					double thetaRad = (theta * CV_PI)/ 180.0;
					int ro = cvRound(x * cos(thetaRad) + y * sin(thetaRad));
					if (ro >= 0 && ro <= D) {
						Hough.at<int>(ro, theta)++;
					}
				}
			}
		}
	}
	 
	double maxHough;
	minMaxLoc(Hough, 0, &maxHough);
	Mat houghImg;
	Hough.convertTo(houghImg, CV_8UC1, 255.0/maxHough);
	imshow("Hough accumulator", houghImg);
	// 2.
	vector<peak> peaks;
	int half  = w / 2;
	for (int ro = half; ro < D - half; ro++) {
		for (int theta = half; theta < 360 - half; theta++) {
			int mid =  Hough.at<int>(ro, theta);
			int ok = 1;
			peak p;
			p.ro = ro;
			p.theta = theta;
			p.hval = mid;
			for (int dy = -half; dy <= half; dy++) {
				for (int dx = -half; dx <= half; dx++) {
					int val = Hough.at<int>(ro + dy, theta + dx);
					if (val > mid) {
						ok = 0;
						break;
					}
				}
				if (ok==0) break;
			}
			if (ok && mid > 0) {
				peaks.push_back(p);
			}
		}
	}
	sort(peaks.begin(),peaks.end());
	if (peaks.size() > k)
		peaks.resize(k);

	// 3. 
	drawHoughLines(img, peaks);
}
//------------------------------|| LAB 4 - Distance Transform ||------------------------------
int weights_di[9] = { -1,-1,-1, 0, 0, 0, 1, 1, 1 };
int weights_dj[9] = { -1, 0, 1,-1, 0, 1,-1, 0, 1 };
int weights[9] = { 3, 2, 3, 2, 0, 2, 3, 2, 3 };
void visualiseImg(Mat img,String s) {
	double minVal, maxVal;
	minMaxLoc(img, &minVal, &maxVal);

	Mat img_vis;
	img.convertTo(img_vis, CV_8UC1, 255.0 / maxVal);

	imshow(s, img_vis);
	waitKey(0);
}
Mat chamfer_dt(Mat& img) {
	int height = img.rows;
	int width = img.cols; 
	const int DT_INF = width * height; 
	Mat dt = Mat(height, width, CV_32SC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == BLACK) { 
				dt.at<int>(i, j) = BLACK;
			}
			else { 
				dt.at<int>(i, j) = WHITE;
			}
		}
	}
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			int x = dt.at<int>(i, j);
			for (int k = 0; k <= 4; k++) {
				x = min(x, dt.at<int>(i + weights_di[k], j + weights_dj[k]) + weights[k]);
			}
			dt.at<int>(i, j) = x;
		}
	}
	for (int i = height - 2; i >= 1; i--) {
		for (int j = width - 2; j >= 1; j--) {
			int x = dt.at<int>(i, j);
			for (int k = 4; k <= 8; k++) { 
				x = min(x, dt.at<int>(i + weights_di[k], j + weights_dj[k]) + weights[k]);
			}
			dt.at<int>(i, j) = x;
		}
	}

	return dt;
}
double matching_score(const Mat& img, const Mat& randomImg) {
	// It s better when the score is smaller
	long long sum = 0;
	int contour_pixels = 0;
	Mat dt = chamfer_dt((Mat)img);
	for (int i = 0; i < randomImg.rows; i++) {
		for (int j = 0; j < randomImg.cols; j++) {
			if (randomImg.at<uchar>(i, j) == BLACK) {
				sum += dt.at<int>(i, j);
				contour_pixels++;
			}
		}
	}
	if (contour_pixels  == 0) {
		printf("Matching Score is 0");
		return 0.0;
	}
	printf("Matching Score : %.4f\n", (double)sum / contour_pixels);
	return (double) sum / contour_pixels;
}
//------------------------------|| LAB 5 - Statistical Data Analysis ||------------------------------
const int FACES = 400;  
const int IMGsize = 19; 
const int NUM = IMGsize * IMGsize;
void linearizeImage(const Mat& img, Mat& featureVector) {
	if (img.empty() || img.rows != IMGsize || img.cols != IMGsize || img.type() != CV_8UC1) {
		featureVector = Mat();
		return;
	}
	featureVector = Mat(1, NUM, CV_32FC1);
	for (int i = 0; i < IMGsize; i++) {
		for (int j = 0; j < IMGsize; j++) {
			featureVector.at<float>(0, i * IMGsize + j) = (float)img.at<uchar>(i, j);
		}
	}
}
Mat load_img() { 
	Mat I(FACES, NUM, CV_32FC1);
	char folder[256] = "Images/faces";
	char fname[256];
	int currentRow = 0; 

	for (int i = 1; i <= FACES; i++) { 
		sprintf(fname, "%s/face%05d.bmp", folder, i); 
		Mat img = imread(fname, 0);

		if (img.empty()) {
			printf("Error: Could not load image %s. Skipping.\n", fname);
			continue;
		}

		Mat featureVector;
		linearizeImage(img, featureVector);

		if (!featureVector.empty()) {
			featureVector.copyTo(I.row(currentRow));
			currentRow++;
		}

		if (i % 50 == 0) {
			printf("Loaded %d images.\n", i);
		}
	}
	 
	return I;
}
void compute_and_save_mean_vector(const Mat& I) {
	if (I.empty()) {
		printf("Error: Feature Matrix I is empty. Cannot compute mean vector.\n");
		return;
	}
 
	Mat mu; 
	cv::reduce(I, mu, 0, REDUCE_AVG, CV_64FC1);
	 
	const char* filename = "mean_values.csv";
	std::ofstream file(filename);

	if (!file.is_open()) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	} 
	for (int j = 0; j < NUM; j++) { 
		double mean_val = mu.at<double>(0, j);
		 
		file << std::fixed << std::setprecision(6) << mean_val;
		 
		if (j < NUM - 1) {
			file << ",";
		}
	}

	file.close();
	printf("Successfully computed mean vector and saved to %s\n", filename);
}
void compute_and_save_covariance_matrix(const Mat& I) {
	if (I.empty()) {
		printf("Error: Feature Matrix I is empty. Cannot compute covariance matrix.\n");
		return;
	}
	 
	Mat mu_vec; 
	cv::reduce(I, mu_vec, 0, REDUCE_AVG, CV_32FC1);  
	Mat I_centered = I.clone();
	for (int i = 0; i < I_centered.rows; i++) { 
		I_centered.row(i) -= mu_vec;
	}
	 
	Mat C = I_centered.t() * I_centered;
	C /= (double)I.rows;  
	 
	const char* filename = "covariance_matrix.csv";
	std::ofstream file(filename);

	if (!file.is_open()) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	}

	printf("Saving Covariance Matrix (size %d x %d) to %s...\n", C.rows, C.cols, filename);
	 
	for (int i = 0; i < C.rows; i++) {
		for (int j = 0; j < C.cols; j++) { 
			double c_val = C.at<float>(i, j);  
			file << std::fixed << std::setprecision(6) << c_val;
			 
			if (j < C.cols - 1) {
				file << ",";
			}
		} 
		file << "\n";
	}

	file.close();
	printf("Successfully computed and saved covariance matrix to %s\n", filename);
}
Mat get_covariance_matrix(const Mat& I) { 
	if (I.empty()) {
		return Mat();
	} 
	Mat mu_vec;
	cv::reduce(I, mu_vec, 0, REDUCE_AVG, CV_32FC1); 
	Mat I_centered = I.clone();
	for (int i = 0; i < I_centered.rows; i++) {
		I_centered.row(i) -= mu_vec;
	} 
	Mat C = I_centered.t() * I_centered;
	C /= (float)I.rows;  

	return C;
}
void compute_and_save_correlation_matrix(const Mat& I) {
	if (I.empty()) {
		printf("Error: Feature Matrix I is empty. Cannot compute correlation matrix.\n");
		return;
	}
	 
	Mat C = get_covariance_matrix(I);
	if (C.empty()) {
		printf("Error: Could not compute covariance matrix C.\n");
		return;
	} 
	Mat sigma(1, NUM, CV_32FC1);
	for (int i = 0; i < NUM; i++) {
		 
		float variance_i = C.at<float>(i, i);
		sigma.at<float>(0, i) = sqrt(variance_i);
	}
	 
	Mat P(NUM, NUM, CV_32FC1);
	for (int i = 0; i < NUM; i++) {
		for (int j = 0; j < NUM; j++) {
			float c_ij = C.at<float>(i, j);
			float sigma_i = sigma.at<float>(0, i);
			float sigma_j = sigma.at<float>(0, j);

			float denominator = sigma_i * sigma_j;
			float rho_ij = 0.0f;

			if (denominator > 1e-6) {  
				rho_ij = c_ij / denominator;
			}
			 
			rho_ij = max(-1.0f, min(1.0f, rho_ij));
			P.at<float>(i, j) = rho_ij;
		}
	} 
	const char* filename = "correlation_matrix.csv";
	std::ofstream file(filename);

	if (!file.is_open()) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	}

	printf("Saving Correlation Matrix (size %d x %d) to %s...\n", P.rows, P.cols, filename);
	 
	for (int i = 0; i < P.rows; i++) {
		for (int j = 0; j < P.cols; j++) {
			float rho_val = P.at<float>(i, j);

			file << std::fixed << std::setprecision(6) << rho_val;

			if (j < P.cols - 1) {
				file << ",";
			}
		}
		file << "\n";
	}

	file.close();
	printf("Successfully computed and saved correlation matrix to %s\n", filename);
}
// --- Task 5 Core Function ---
int get_linear_index(int row, int col) {
	if (row < 0 || row >= IMGsize || col < 0 || col >= IMGsize) {
		return -1; 
	}
	return row * IMGsize + col;
}
void analyze_feature_pair(const Mat& I, const Mat& C, int row1, int col1, int row2, int col2, const char* description) { 
	int i = get_linear_index(row1, col1);
	int j = get_linear_index(row2, col2);

	if (i == -1 || j == -1) {
		printf("Error: Invalid coordinate pair (%d,%d) or (%d,%d). Index out of bounds.\\n", row1, col1, row2, col2);
		return;
	} 
	float c_ij = C.at<float>(i, j);
	float c_ii = C.at<float>(i, i); 
	float c_jj = C.at<float>(j, j);  

	float sigma_i = sqrt(c_ii);
	float sigma_j = sqrt(c_jj);
	 
	float rho_ij = 0.0f;
	float denominator = sigma_i * sigma_j;
	if (denominator > 1e-6) {
		rho_ij = c_ij / denominator;
	}
	rho_ij = max(-1.0f, min(1.0f, rho_ij)); 

	printf("\n--- %s ---\n", description);
	printf("Feature 1 (row=%d, col=%d) -> Index i=%d\n", row1, col1, i);
	printf("Feature 2 (row=%d, col=%d) -> Index j=%d\n", row2, col2, j);
	printf("Correlation Coefficient: %.2f\n", rho_ij);

	 
	const int CHART_SIZE = 256;
	Mat chart(CHART_SIZE, CHART_SIZE, CV_8UC1, Scalar(255));  
	 
	for (int k = 0; k < I.rows; k++) { 
		int x = (int)round(I.at<float>(k, j));
		int y = (int)round(I.at<float>(k, i));
		 
		if (x >= 0 && x < CHART_SIZE && y >= 0 && y < CHART_SIZE) {
			chart.at<uchar>(y, x) = 0;
		}
	}

	imshow(description, chart);
	 
} 
void task5_analyze_all_pairs(const Mat& I) { 
	Mat C = get_covariance_matrix(I);
	if (C.empty()) return; 
	analyze_feature_pair(I, C, 5, 4, 5, 14, "Task 5a (Eyes) rho ~ 0.94");
	 
	analyze_feature_pair(I, C, 10, 3, 9, 15, "Task 5b (Cheeks) rho ~ 0.84");
	 
	analyze_feature_pair(I, C, 5, 4, 18, 0, "Task 5c (Eye/Corner) rho ~ 0.07");
}

 
int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);
	// ----- Lab 3
	/*Mat img = imread("Images/edge_simple.bmp", IMREAD_GRAYSCALE);
	hough(img, 3, 6);
	waitKey(0); 
	return 0;*/

	// ----- Lab 4
	/*Mat img = imread("Images/template.bmp", IMREAD_GRAYSCALE);
	Mat random =   imread("Images/unknown_object1.bmp", IMREAD_GRAYSCALE);
	Mat random1 = imread("Images/unknown_object2.bmp", IMREAD_GRAYSCALE);
	imshow("Original", img);
	Mat dt = chamfer_dt(img);
	visualiseImg(dt, "DT - Normalized");
	double s1 = matching_score(img, random);
	double s2 = matching_score(img, random1);*/
	
	// ----- Lab 5
	Mat I = load_img();
	compute_and_save_mean_vector(I);
	compute_and_save_covariance_matrix(I);
	compute_and_save_correlation_matrix(I);
	task5_analyze_all_pairs(I);

	waitKey(0);
	return 0;

}
