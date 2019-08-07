////***********API functions
//// file I/O
//extern "C"
char* concat(int count, ...);
//extern "C"
bool fexists(const char * filename);
//extern "C"
unsigned short gettifinfo(char tifdir[], unsigned int *tifSize);
//extern "C"
void readtifstack(float *h_Image, char tifdir[], unsigned int *imsize);
//extern "C"
void writetifstack(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample);

void readtifstack_16to16(unsigned short *h_Image, char tifdir[], unsigned int *imsize);

void writetifstack_16to16(char tifdir[], unsigned short *h_Image, unsigned int *imsize);

// 
void queryDevice();

//// 2D registration
int reg_2dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float FTOL, int itLimit, int deviceNum, float *regRecords);

int reg_2dshiftaligngpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float shiftRegion, int totalStep, int deviceNum, float *regRecords);

int reg_2dshiftalignXgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float shiftRegion, int totalStep, int deviceNum, float *regRecords);

//// 3D registration
int reg_3dcpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
int inputTmx, float FTOL, int itLimit, int subBgTrigger, float *regRecords);

int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
	int inputTmx, float FTOL, int itLimit, int flagSubBg, int deviceNum, float *regRecords);

int reg_3dphasetransgpu(int *shiftXYZ, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int downSample, int deviceNum, float *regRecords);

int affinetrans_3dgpu(float *h_reg, float *iTmx, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);
int affinetrans_3dgpu_16to16(unsigned short *h_reg, float *iTmx, unsigned short *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);
//// 3D deconvolution
// single view 
int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize,
	int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords, bool flagUnmatch, float *h_psf_bp);
// dual view
int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2,
	unsigned int *psfSize, int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);

//// 3D fusion: dual view registration and deconvolution
int fusion_dualview(float *h_decon, float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2,
	float *pixelSize1, float *pixelSize2, int imRotation, int regMethod, int flagInitialTmx, float FTOL, int itLimit, float *h_psf1, float *h_psf2,
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);
//// maximum intensity projections: 
int mp2Dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj);
int mp3Dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum);

//// 3D registration and deconvolution: batch processing
int reg_3dgpu_batch(char *outMainFolder, char *folder1, char *folder2, char *fileNamePrefix1, char *fileNamePrefix2, int imgNumStart, int imgNumEnd, int imgNumInterval, int imgNumTest,
	float *pixelSize1, float *pixelSize2, int regMode, int imRotation, int flagInitialTmx, float *iTmx, float FTOL, int itLimit, int deviceNum, int *flagSaveInterFiles, float *records);

int decon_singleview_batch(char *outMainFolder, char *folder, char *fileNamePrefix, int imgNumStart, int imgNumEnd, int imgNumInterval, char *filePSF,
	int itNumForDecon, int deviceNum, int bitPerSample, bool flagMultiColor, float *records, bool flagUnmatch, char *filePSF_bp);

int fusion_dualview_batch(char *outFolder, char *inFolder1, char *inFolder2, char *fileNamePrefix1, char *fileNamePrefix2, int imgNumStart, int imgNumEnd, int imgNumInterval, int imgNumTest,
float *pixelSize1, float *pixelSize2, int regMode, int imRotation, int flagInitialTmx, float *iTmx, float FTOL, int itLimit, char *filePSF1, char *filePSF2,
int itNumForDecon, int deviceNum, int *flagSaveInterFiles, int bitPerSample, float *records, bool flagUnmatch, char *filePSF_bp1, char *filePSF_bp2);
