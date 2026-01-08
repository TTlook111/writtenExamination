<template>
  <div class="upload-resume-container">
    <el-row :gutter="40" class="main-row">
      <!-- Left Side: Illustration and Text -->
      <el-col :xs="24" :sm="24" :md="12" :lg="12" class="left-col">
        <div class="left-content">
          <h1 class="main-title">Finish!</h1>
          <h2 class="sub-title">Upload Your Resume</h2>
          <p class="description">
            Upload your resume, the platform will help you parse and optimize, you can confirm in this step
          </p>
          
          <div class="illustration-placeholder">
             <div class="step-badge">
                <span class="step-num">1</span>
                <div class="step-avatar">朱</div>
             </div>
             <div class="cat-illustration">
               <img :src="catImage" alt="Cat Illustration" style="max-width: 100%; height: auto;" />
            </div>
         </div>
       </div>
     </el-col>

     <!-- Right Side: Upload Card -->
     <el-col :xs="24" :sm="24" :md="12" :lg="12" class="right-col">
        <div class="upload-card">
          <h3 class="card-title">Upload file</h3>
          
          <el-upload
            class="upload-area"
            drag
            action="#"
            :auto-upload="false"
            :on-change="handleChange"
            :show-file-list="false"
            multiple
          >
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              <template v-if="fileName">
                Selected file: <strong>{{ fileName }}</strong>
              </template>
              <template v-else>
                Drag your resume file to this area, or click on the area to select<br>
                the appropriate file to upload
              </template>
            </div>
          </el-upload>

          <div class="actions">
            <el-button class="btn-last-step">Last step</el-button>
            <el-button type="primary" class="btn-finish">Finish</el-button>
          </div>

          <div class="steps-indicator">
            <div class="custom-steps">
               <div class="step-item completed">
                  <div class="step-circle"><el-icon><Check /></el-icon></div>
                  <div class="step-line"></div>
               </div>
               <div class="step-item completed">
                  <div class="step-circle"><el-icon><Check /></el-icon></div>
                  <div class="step-line"></div>
               </div>
               <div class="step-item active">
                  <div class="step-circle">3</div>
               </div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { UploadFilled, Check } from '@element-plus/icons-vue'
import catImage from '@/assets/image.png'

const fileName = ref('')

const handleChange = (uploadFile) => {
  fileName.value = uploadFile.name
}
</script>

<style scoped>
.upload-resume-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
  background-color: #ffffff; /* Main background white as per image */
  min-height: 100vh;
  display: flex;
  align-items: center;
}

.main-row {
  width: 100%;
  align-items: center;
}

/* Left Column Styles */
.left-content {
  padding: 20px;
  text-align: left;
}

.main-title {
  font-size: 3.5rem; /* Increased size */
  font-weight: 700;
  color: #1a1a1a;
  margin-bottom: 0.5rem;
  line-height: 1.2;
}

.sub-title {
  font-size: 2.5rem; /* Increased size */
  font-weight: 500;
  color: #1a1a1a;
  margin-bottom: 1.5rem;
  line-height: 1.2;
}

.description {
  color: #a0a0a0;
  font-size: 1.1rem;
  line-height: 1.6;
  max-width: 450px;
  margin-bottom: 3rem;
}

.illustration-placeholder {
  position: relative;
  margin-top: 2rem;
  display: inline-block; /* Allow badge to position relative to this */
}

.step-badge {
  position: absolute;
  top: -10px;
  left: -20px;
  background: white;
  border-radius: 30px;
  padding: 8px 12px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  display: flex;
  align-items: center;
  gap: 8px;
  z-index: 2;
}

.step-num {
  background: #5c6b8b;
  color: white;
  border-radius: 50%;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: bold;
}

.step-avatar {
  background: #007bff;
  color: white;
  border-radius: 50%;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  font-weight: bold;
}

/* Right Column Styles */
.upload-card {
  background: white;
  border-radius: 24px;
  padding: 50px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.08); /* Softer, larger shadow */
  max-width: 550px;
  margin: 0 auto;
}

.card-title {
  font-size: 1.8rem;
  margin-bottom: 2rem;
  color: #1a1a1a;
  text-align: center; /* Centered as per image usually, or left? Image shows left aligned 'Upload file' actually? Wait, image shows 'Upload file' centered in the card header area visually? No, it looks slightly left or center. Let's stick to center for card header vibes or check image again. Image: 'Upload file' is clearly Left aligned in the second crop, but centered in the first crop? Actually it's Left aligned in the second crop. */
  text-align: left;
  font-weight: 600;
}

.upload-area {
  margin-bottom: 3rem;
}

:deep(.el-upload-dragger) {
  background-color: #f2f8fe; /* Very light blue */
  border: 2px dashed #a0cfff; /* Light blue dashed border */
  border-radius: 16px;
  height: 250px; /* Taller */
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  transition: all 0.3s;
}

:deep(.el-upload-dragger:hover) {
  border-color: #2b65ff;
  background-color: #e6f0ff;
}

:deep(.el-icon--upload) {
  font-size: 64px; /* Larger icon */
  color: #2b65ff; /* Stronger blue */
  margin-bottom: 20px;
}

.el-upload__text {
  color: #b0b0b0;
  font-size: 14px;
  padding: 0 40px;
  line-height: 1.6;
  text-align: center;
}

.actions {
  display: flex;
  justify-content: space-between;
  margin-bottom: 3rem;
  gap: 30px;
}

.btn-last-step {
  flex: 1;
  background-color: #8bb4ff;
  border-color: #8bb4ff;
  color: white;
  border-radius: 8px;
  height: 50px;
  font-size: 16px;
  font-weight: 500;
}
.btn-last-step:hover {
  background-color: #7aa3ee;
  border-color: #7aa3ee;
  color: white;
}

.btn-finish {
  flex: 1;
  background-color: #1a54ec; /* Vivid blue */
  border-color: #1a54ec;
  border-radius: 8px;
  height: 50px;
  font-size: 16px;
  font-weight: 500;
  transition: all 0.3s;
}
.btn-finish:hover {
  background-color: #1040b0;
  border-color: #1040b0;
}

/* Custom Steps Indicator */
.custom-steps {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.step-item {
  display: flex;
  align-items: center;
}

.step-circle {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  color: white;
  background-color: #e0e0e0; /* Default gray */
}

.step-line {
  width: 60px;
  height: 2px;
  background-color: #e0e0e0;
  margin-left: 10px;
}

.step-item.completed .step-circle {
  background-color: #6699ff; /* Light blue check */
}

.step-item.completed .step-line {
  background-color: #6699ff;
}

.step-item.active .step-circle {
  background-color: #6699ff; /* Active step blue */
  width: 24px;
  height: 24px;
  font-weight: bold;
}

/* Responsive Adjustments */
@media (max-width: 992px) {
  .upload-resume-container {
    padding: 20px;
    flex-direction: column;
    height: auto;
  }
  
  .left-content {
    text-align: center;
    margin-bottom: 40px;
    padding: 0;
  }
  
  .description {
    margin: 0 auto 2rem;
  }
  
  .illustration-placeholder {
    display: inline-block;
  }

  .step-badge {
    left: 50%;
    transform: translateX(-150%); /* Adjust based on illustration */
  }
}
</style>
