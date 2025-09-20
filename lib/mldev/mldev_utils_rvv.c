#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "mldev_utils.h"
#include "net/rte_ether.h"

#include <riscv_vector.h>

#include <eal_export.h>
#include <sys/types.h>

/* Description:
 * This file implements vector versions of Machine Learning utility functions used to convert data
 * types from higher precision to lower precision and vice-versa, except bfloat16. Implementation
 * is based on Arm Neon intrinsics.
 */

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_int8, 22.11)
int
rte_ml_io_float32_to_int8(const void *input, void *output, uint64_t nb_elements, float scale,
			  int8_t zero_point)
{
	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;

    size_t vl = 0;
    for (size_t i = 0; i < nb_elements; i += vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        // 加载float32位类型向量,按照最低128位来，可以加载到8个元素
        vfloat32m4_t vf32m4_input = __riscv_vle32_v_f32m4((const float*)((const int8_t*)input+i*4),vl);
        vfloat32m4_t vf32m4_input_scale = __riscv_vfdiv_vf_f32m4(vf32m4_input, scale, vl);
        // 添加零点偏移
        vfloat32m4_t vf32m4_input_scale_zeropoint = __riscv_vfadd_vf_f32m4(vf32m4_input_scale, (float)zero_point, vl);
        // vint32m4_t vi32m4_convert_input = __riscv_vfcvt_x_f_v_i32m4(vf32m4_input_scale_zeropoint, vl);
        // 使用正确的舍入模式进行转换 (RMM = 远离零舍入)
        vint32m4_t vi32m4_convert_input = __riscv_vfcvt_x_f_v_i32m4_rm(
            vf32m4_input_scale_zeropoint, 
            __RISCV_FRM_RMM, 
            vl
        );
        // 设置饱和边界
        vint32m4_t min_val = __riscv_vmv_v_x_i32m4(INT8_MIN, vl);
        vint32m4_t max_val = __riscv_vmv_v_x_i32m4(INT8_MAX, vl);
        // 应用饱和限制
        vi32m4_convert_input = __riscv_vmax_vv_i32m4(vi32m4_convert_input, min_val, vl);
        vi32m4_convert_input = __riscv_vmin_vv_i32m4(vi32m4_convert_input, max_val, vl);
        vint16m2_t vi16m2_clip_inpit = __riscv_vnclip_wx_i16m2(vi32m4_convert_input, 0, __RISCV_VXRM_RNU, vl);
        vint8m1_t vi8m1_clip_inpit = __riscv_vnclip_wx_i8m1(vi16m2_clip_inpit, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8m1((int8_t*)(output+i), vi8m1_clip_inpit, vl);
    }
	return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_int8_to_float32, 22.11)
int rte_ml_io_int8_to_float32(const void *input, void *output, uint64_t nb_elements, float scale,
                             int8_t zero_point)
{
    if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;

    const int8_t *input_buffer = (const int8_t *)input;
    float *output_buffer = (float *)output;

    size_t vl;
    for (size_t i = 0; i < nb_elements; i += vl) {
        // 1. 设置向量长度（基于剩余元素）
        vl = __riscv_vsetvl_e8m1(nb_elements - i);
        
        // 2. 加载 int8 向量
        vint8m1_t v_i8 = __riscv_vle8_v_i8m1(input_buffer + i, vl);
        
        // 3. 将 int8 扩展为 int32（符号扩展）
        vint32m4_t v_i32 = __riscv_vwadd_vx_i32m4(
            __riscv_vwadd_vx_i16m2(v_i8, 0, vl),  // 先扩展到 int16
            0, 
            vl
        );
        
        // 4. 将 int32 转换为 float32
        vfloat32m4_t v_f32 = __riscv_vfcvt_f_x_v_f32m4(v_i32, vl);
        
        // 5. 减去零点并应用缩放
        v_f32 = __riscv_vfsub_vf_f32m4(v_f32, (float)zero_point, vl);
        v_f32 = __riscv_vfmul_vf_f32m4(v_f32, scale, vl);
        
        // 6. 存储 float32 结果
        __riscv_vse32_v_f32m4(output_buffer + i, v_f32, vl);
    }
    return 0;
}


RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_uint8, 22.11)
int
rte_ml_io_float32_to_uint8(const void *input, void *output, uint64_t nb_elements, float scale,
			   uint8_t zero_point)
{
    if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;

    size_t vl = 0;
    for (size_t i=0; i<nb_elements; i+=vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        // 加载float32位类型向量,按照最低128位来，可以加载到8个元素
        vfloat32m4_t vf32m4_input = __riscv_vle32_v_f32m4((const float*)((const uint8_t*)input+i*4),vl);
        vfloat32m4_t vf32m4_input_scale = __riscv_vfdiv_vf_f32m4(vf32m4_input, scale, vl);
        // 添加零点偏移
        vfloat32m4_t vf32m4_input_scale_zeropoint = __riscv_vfadd_vf_f32m4(vf32m4_input_scale, (float)zero_point, vl);
        vuint32m4_t vu32m4_convert_input = __riscv_vfcvt_xu_f_v_u32m4_rm(
            vf32m4_input_scale_zeropoint, 
            __RISCV_FRM_RMM, 
            vl
        );
        // 设置饱和边界
        vuint32m4_t min_val = __riscv_vmv_v_x_u32m4(0, vl);
        vuint32m4_t max_val = __riscv_vmv_v_x_u32m4(UINT8_MAX, vl);
        // 应用饱和限制
        vu32m4_convert_input = __riscv_vmaxu_vv_u32m4(vu32m4_convert_input, min_val, vl);
        vu32m4_convert_input = __riscv_vminu_vv_u32m4(vu32m4_convert_input, max_val, vl);
        vuint16m2_t vu16m2_clip_inpit = __riscv_vnclipu_wx_u16m2(vu32m4_convert_input, 0, __RISCV_VXRM_RNU, vl);
        vuint8m1_t vu8m1_clip_inpit = __riscv_vnclipu_wx_u8m1(vu16m2_clip_inpit, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_u8m1((uint8_t*)(output+i), vu8m1_clip_inpit, vl);
    }
    return 0;
}


RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_uint8_to_float32, 22.11)
int
rte_ml_io_uint8_to_float32(const void *input, void *output, uint64_t nb_elements, float scale,
                           uint8_t zero_point)
{
    if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;
    
    size_t vl = 0;
    const uint8_t *in = (const uint8_t *)input;
    float *out = (float *)output;

    for (size_t i = 0; i < nb_elements; i += vl) {
        vl = __riscv_vsetvl_e8m1(nb_elements - i);
        
        // 1. 加载uint8数据
        vuint8m1_t vu8_input = __riscv_vle8_v_u8m1(in + i, vl);
        
        // 2. 减去zero_point（结果可能为负，所以需要转为有符号）
        vint16m2_t vi16_diff = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwaddu_vx_u16m2(vu8_input, 0, vl));
        vint16m2_t vzero = __riscv_vmv_v_x_i16m2((int16_t)zero_point, vl);
        vint16m2_t vsub = __riscv_vsub_vv_i16m2(vi16_diff, vzero, vl);
        
        // 3. 扩展为32位有符号整数
        vint32m4_t vint32 = __riscv_vsext_vf2_i32m4(vsub, vl);
        
        // 4. 转换为浮点数（使用有符号转换）
        vfloat32m4_t vfloat = __riscv_vfcvt_f_x_v_f32m4_rm(vint32, __RISCV_FRM_RMM, vl);
        
        // 5. 乘以scale（正确的参数）
        vfloat = __riscv_vfmul_vf_f32m4(vfloat, scale, vl);
        
        // 6. 存储结果
        __riscv_vse32_v_f32m4(out + i, vfloat, vl);
    }
    
    return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_int16, 22.11)
int
rte_ml_io_float32_to_int16(const void *input, void *output, uint64_t nb_elements, float scale,
			   int16_t zero_point)
{
	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;
	size_t vl = 0;
    for (size_t i = 0; i < nb_elements; i += vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        // 加载float32位类型向量,按照最低128位来，可以加载到8个元素
        vfloat32m4_t vf32m4_input = __riscv_vle32_v_f32m4((const float*)(input)+i,vl);
        vfloat32m4_t vf32m4_input_scale = __riscv_vfdiv_vf_f32m4(vf32m4_input, scale, vl);
        // 添加零点偏移
        vfloat32m4_t vf32m4_input_scale_zeropoint = __riscv_vfadd_vf_f32m4(vf32m4_input_scale, (float)zero_point, vl);
        vint32m4_t vi32m4_convert_input = __riscv_vfcvt_x_f_v_i32m4_rm(vf32m4_input_scale_zeropoint,__RISCV_FRM_RMM, vl);
        // 设置饱和边界
        vint32m4_t min_val = __riscv_vmv_v_x_i32m4(INT16_MIN, vl);
        vint32m4_t max_val = __riscv_vmv_v_x_i32m4(INT16_MAX, vl);
        // 应用饱和限制
        vi32m4_convert_input = __riscv_vmax_vv_i32m4(vi32m4_convert_input, min_val, vl);
        vi32m4_convert_input = __riscv_vmin_vv_i32m4(vi32m4_convert_input, max_val, vl);
        vint16m2_t vi16m2_clip_inpit = __riscv_vnclip_wx_i16m2(vi32m4_convert_input, 0, __RISCV_VXRM_RNU, vl);
        // vint8m1_t vi8m1_clip_inpit = __riscv_vnclip_wx_i8m1(vi16m2_clip_inpit, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse16_v_i16m2((int16_t*)(output)+i, vi16m2_clip_inpit, vl);
    }
	return 0;
}


RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_int16_to_float32, 22.11)
int
rte_ml_io_int16_to_float32(const void *input, void *output, uint64_t nb_elements, float scale, int16_t zero_point)
{
	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;
    size_t vl = 0;
    for (size_t i=0; i<nb_elements; i+=vl) {
        vl = __riscv_vsetvl_e16m2(nb_elements-i);
        vint16m2_t vi16_input = __riscv_vle16_v_i16m2((const int16_t*)(input)+i, vl);
        vint32m4_t vi32m4_convert_input = __riscv_vsext_vf2_i32m4(vi16_input, vl);
        vint32m4_t vi32m4_input_zeropoint = __riscv_vsub_vx_i32m4(vi32m4_convert_input, (int32_t)zero_point, vl);
		vfloat32m4_t vf32m4_convert_input = __riscv_vfcvt_f_x_v_f32m4_rm(vi32m4_input_zeropoint, __RISCV_FRM_RMM, vl);

		vf32m4_convert_input = __riscv_vfmul_vf_f32m4(vf32m4_convert_input, scale, vl);

		__riscv_vse32_v_f32m4((float*)(output)+i, vf32m4_convert_input, vl);
    }
	return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_uint16, 22.11)
int
rte_ml_io_float32_to_uint16(const void *input, void *output, uint64_t nb_elements, float scale,
			    uint16_t zero_point)
{
	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;

	size_t vl = 0;
    for (size_t i = 0; i < nb_elements; i += vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        // 加载float32位类型向量,按照最低128位来，可以加载到8个元素
        vfloat32m4_t vf32m4_input = __riscv_vle32_v_f32m4(((const float*)input)+i,vl);
        vfloat32m4_t vf32m4_input_scale = __riscv_vfdiv_vf_f32m4(vf32m4_input, scale, vl);
        // 添加零点偏移
        vfloat32m4_t vf32m4_input_scale_zeropoint = __riscv_vfadd_vf_f32m4(vf32m4_input_scale, (float)zero_point, vl);
        vuint32m4_t vu32m4_convert_input = __riscv_vfcvt_xu_f_v_u32m4_rm(vf32m4_input_scale_zeropoint, __RISCV_FRM_RMM, vl);
        // 设置饱和边界
        vuint32m4_t min_val = __riscv_vmv_v_x_u32m4(0, vl);
        vuint32m4_t max_val = __riscv_vmv_v_x_u32m4(UINT16_MAX, vl);
        // 应用饱和限制
        vu32m4_convert_input = __riscv_vmaxu_vv_u32m4(vu32m4_convert_input, min_val, vl);
        vu32m4_convert_input = __riscv_vminu_vv_u32m4(vu32m4_convert_input, max_val, vl);
        vuint16m2_t vi16m2_clip_inpit = __riscv_vnclipu_wx_u16m2(vu32m4_convert_input, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse16_v_u16m2(((int16_t*)output)+i, vi16m2_clip_inpit, vl);
    }
	return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_uint16_to_float32, 22.11)
int
rte_ml_io_uint16_to_float32(const void *input, void *output, uint64_t nb_elements, float scale, uint16_t zero_point)
{
	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;

    size_t vl = 0;
    for (size_t i=0; i<nb_elements; i+=vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        vuint16m2_t vu16m2_input = __riscv_vle16_v_u16m2(((const uint16_t*)input)+i, vl);
        // 拓宽为32位有符号数
        vint32m4_t vi32m4_convert_input = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwaddu_vx_u32m4(vu16m2_input, 0, vl));
        // 减去了零点偏移
        vint32m4_t vi32m4_sub_input = __riscv_vsub_vx_i32m4(vi32m4_convert_input, (int32_t)zero_point, vl);
        // 转位float32类型
        vfloat32m4_t vf32m4_convert = __riscv_vfcvt_f_x_v_f32m4_rm(vi32m4_sub_input, __RISCV_FRM_RMM, vl);

        vf32m4_convert = __riscv_vfmul_vf_f32m4(vf32m4_convert, scale, vl); 
        // 存储内存
        __riscv_vse32_v_f32m4(((float*)output)+i, vf32m4_convert, vl);

    }
	return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_int32, 22.11)
int
rte_ml_io_float32_to_int32(const void *input, void *output, uint64_t nb_elements, float scale,
			   int32_t zero_point)
{
	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;
    size_t vl = 0;
    for (size_t i=0; i<nb_elements; i+=vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        vfloat32m4_t vf32m4_input = __riscv_vle32_v_f32m4(((const float*)input) + i, vl);
        vfloat32m4_t vf32m4_scale_input = __riscv_vfdiv_vf_f32m4(vf32m4_input, scale, vl);
        // 在浮点域添加零点偏移
        vfloat32m4_t vf32m4_with_zp = __riscv_vfadd_vf_f32m4(vf32m4_scale_input, (float)zero_point, vl);

        // 转为int32位
        vint32m4_t vi32m4_input = __riscv_vfcvt_x_f_v_i32m4_rm(vf32m4_with_zp, __RISCV_FRM_RMM, vl);
        // 偏移0点
        // vint32m4_t vi32m4_add_input = __riscv_vadd_vx_i32m4(vi32m4_input, zero_point, vl);
        vint32m4_t vi32m4_saturated = __riscv_vmax_vx_i32m4(vi32m4_input, INT32_MIN, vl);
        vi32m4_saturated = __riscv_vmin_vx_i32m4(vi32m4_saturated, INT32_MAX, vl);
        __riscv_vse32_v_i32m4(((int32_t*)output)+i, vi32m4_saturated, vl);
    }
	return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_int32_to_float32, 22.11)
int
rte_ml_io_int32_to_float32(const void *input, void *output, uint64_t nb_elements, float scale, int32_t zero_point)
{
	const int32_t *input_buffer;
	float *output_buffer;
	uint64_t i;

	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;
    size_t vl = 0;
    for (size_t i = 0; i < nb_elements; i += vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements - i);
        vint32m4_t vi32m4_input = __riscv_vle32_v_i32m4(((const int32_t*)input)+i, vl);
        vint32m4_t vi32m4_sub_input = __riscv_vsub_vx_i32m4(vi32m4_input, zero_point, vl);
        vfloat32m4_t vf32m4_input =  __riscv_vfcvt_f_x_v_f32m4_rm(vi32m4_sub_input,__RISCV_FRM_RMM, vl);
        vf32m4_input = __riscv_vfmul_vf_f32m4(vf32m4_input, scale, vl);
        __riscv_vse32_v_f32m4((float*)output+i, vf32m4_input, vl);

    }
	return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_uint32, 22.11)
int
rte_ml_io_float32_to_uint32(const void *input, void *output, uint64_t nb_elements, float scale,
			    uint32_t zero_point)
{
	const float *input_buffer;
	uint32_t *output_buffer;
	int32_t i32;
	uint64_t i;

	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;

	size_t vl = 0;
    for (size_t i=0; i<nb_elements; i+=vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        vfloat32m4_t vf32m4_input = __riscv_vle32_v_f32m4(((const float*)input) + i, vl);
        vfloat32m4_t vf32m4_scale_input = __riscv_vfdiv_vf_f32m4(vf32m4_input, scale, vl);
        // 在浮点域添加零点偏移
        vfloat32m4_t vf32m4_with_zp = __riscv_vfadd_vf_f32m4(vf32m4_scale_input, (float)zero_point, vl);

        // 转为int32位
        vuint32m4_t vi32m4_input = __riscv_vfcvt_xu_f_v_u32m4_rm(vf32m4_with_zp, __RISCV_FRM_RMM, vl);
        // 偏移0点
        vuint32m4_t vi32m4_saturated = __riscv_vmaxu_vx_u32m4(vi32m4_input, 0, vl);
        vi32m4_saturated = __riscv_vminu_vx_u32m4(vi32m4_saturated, UINT32_MAX, vl);
        __riscv_vse32_v_u32m4(((uint32_t*)output)+i, vi32m4_saturated, vl);
    }
	return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_uint32_to_float32, 22.11)
int
rte_ml_io_uint32_to_float32(const void *input, void *output, uint64_t nb_elements, float scale, uint32_t zero_point)
{
	const uint32_t *input_buffer;
	float *output_buffer;
	uint64_t i;

	if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
		return -EINVAL;

	input_buffer = (const uint32_t *)input;
	output_buffer = (float *)output;

	size_t vl = 0;
    for (size_t i=0; i<nb_elements; i+=vl) {
        vl = __riscv_vsetvl_e32m4(nb_elements-i);
        vuint32m4_t vu32m4_input = __riscv_vle32_v_u32m4(input_buffer+i, vl);
        vuint32m4_t vint32m4_sub = __riscv_vsub_vx_u32m4(
            vu32m4_input,
            zero_point,
            vl
        );
        vfloat32m4_t vf32m4_convert_input = __riscv_vfcvt_f_xu_v_f32m4_rm(vint32m4_sub, __RISCV_FRM_RMM, vl);
        vfloat32m4_t vf32m4_output = __riscv_vfmul_vf_f32m4(vf32m4_convert_input, scale, vl);
        __riscv_vse32_v_f32m4(output_buffer+i, vf32m4_output, vl);
    }
	return 0;
}


RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_int64, 22.11)
int rte_ml_io_float32_to_int64(const void *input, void *output, uint64_t nb_elements, 
                              float scale, int64_t zero_point)
{
    // 参数检查
    if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;

    const float *input_buffer = (const float *)input;
    int64_t *output_buffer = (int64_t *)output;

    // 提前计算倒数缩放因子 (避免循环内重复计算)
    const float inv_scale = 1.0f / scale;
    // 注意：RVV需要double类型进行64位整数转换
    const double inv_scale_double = (double)inv_scale;

    size_t vl;
    for (size_t i = 0; i < nb_elements; i += vl) {
        // 1. 设置向量长度 (基于float32元素)
        vl = __riscv_vsetvl_e32m2(nb_elements - i);
        
        // 2. 加载float32向量
        vfloat32m2_t v_f32 = __riscv_vle32_v_f32m2(input_buffer + i, vl);
        
        // 3. 扩展float32到float64 (支持64位整数转换)
        vfloat64m4_t v_f64 = __riscv_vfwcvt_f_f_v_f64m4(v_f32, vl);
        
        // 4. 应用缩放
        vfloat64m4_t v_scaled = __riscv_vfmul_vf_f64m4(v_f64, inv_scale_double, vl);
        
        // 5. 四舍五入到最近整数 (使用RMM舍入模式)
        // vfloat64m4_t v_rounded = __riscv_vfrround_vf_f64m4(v_scaled, vl, __RISCV_FRM_RNE);
        
        // 6. 转换为int64 (饱和转换防止溢出)
        vint64m4_t v_int = __riscv_vfcvt_x_f_v_i64m4_rm(v_scaled, __RISCV_FRM_RMM, vl);
        
        // 7. 添加零点偏移
        vint64m4_t v_result = __riscv_vadd_vx_i64m4(v_int, zero_point, vl);
        
        // 8. 存储int64结果
        __riscv_vse64_v_i64m4(output_buffer + i, v_result, vl);
    }
    return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_int64_to_float32, 22.11)
int rte_ml_io_int64_to_float32(const void *input, void *output, uint64_t nb_elements,
                              float scale, int64_t zero_point)
{
    // 参数检查
    if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;

    const int64_t *input_buffer = (const int64_t *)input;
    float *output_buffer = (float *)output;

    const double scale_double = (double)scale;
    
    size_t vl;
    for (size_t i = 0; i < nb_elements; i += vl) {
        // 1. 设置向量长度（基于int64元素）
        vl = __riscv_vsetvl_e64m2(nb_elements - i);
        
        // 2. 加载int64向量
        vint64m2_t v_i64 = __riscv_vle64_v_i64m2(input_buffer + i, vl);
        
        // 3. 减去零点（zero_point）
        vint64m2_t v_offset = __riscv_vsub_vx_i64m2(v_i64, zero_point, vl);
        
        // 4. 将int64转换为double（保持精度）
        vfloat64m2_t v_f64 = __riscv_vfcvt_f_x_v_f64m2(v_offset, vl);
        
        // 5. 应用缩放（使用double精度）
        vfloat64m2_t v_scaled = __riscv_vfmul_vf_f64m2(v_f64, scale_double, vl);
        
        // 6. 将double转换为float32（窄化转换）
        vfloat32m1_t v_f32 = __riscv_vfncvt_f_f_w_f32m1_rm(v_scaled, __RISCV_FRM_RMM, vl);
        
        // 7. 存储float32结果
        __riscv_vse32_v_f32m1(output_buffer + i, v_f32, vl);
    }
    return 0;
}


RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_uint64, 22.11)
int rte_ml_io_float32_to_uint64(const void *input, void *output, uint64_t nb_elements,
                               float scale, uint64_t zero_point)
{
    // 参数检查
    if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;

    const float *input_buffer = (const float *)input;
    uint64_t *output_buffer = (uint64_t *)output;

    // 优化：使用双精度计算避免精度损失
    const double inv_scale = 1.0 / (double)scale;  // 倒数缩放因子
    const double zero_point_d = (double)zero_point; // 零点转换为double

    size_t vl;
    for (size_t i = 0; i < nb_elements; i += vl) {
        // 1. 设置向量长度（基于float32元素）
        vl = __riscv_vsetvl_e32m2(nb_elements - i);
        
        // 2. 加载float32向量
        vfloat32m2_t v_f32 = __riscv_vle32_v_f32m2(input_buffer + i, vl);
        
        // 3. 扩展float32到float64（保持精度）
        vfloat64m4_t v_f64 = __riscv_vfwcvt_f_f_v_f64m4(v_f32, vl);
        
        // 4. 应用缩放（乘以倒数避免除法）
        vfloat64m4_t v_scaled = __riscv_vfmul_vf_f64m4(v_f64, inv_scale, vl);
        
        // // 5. 四舍五入到最近整数（使用RNE舍入模式）
        // vfloat64m4_t v_rounded = __riscv_vfrround_vf_f64m4(v_scaled, __RISCV_FRM_RNE, vl);
        
        // 6. 转换为int64（有符号整数）
        vuint64m4_t v_i64 = __riscv_vfcvt_xu_f_v_u64m4_rm(v_scaled, __RISCV_FRM_RMM, vl);
        
        // 7. 添加零点偏移（转换为有符号加法）
        vuint64m4_t v_offset = __riscv_vsaddu_vx_u64m4(v_i64, (uint64_t)zero_point, vl);

        // 9. 存储为uint64（直接转换位模式）
        __riscv_vse64_v_u64m4(output_buffer + i, v_offset, vl);
    }
    return 0;
}


RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_uint64_to_float32, 22.11)
int rte_ml_io_uint64_to_float32(const void *input, void *output, uint64_t nb_elements,
                               float scale, uint64_t zero_point)
{
    // 参数检查
    if ((scale == 0) || (nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;

    const uint64_t *input_buffer = (const uint64_t *)input;
    float *output_buffer = (float *)output;

    // 优化：使用双精度缩放避免精度损失
    const double scale_double = (double)scale;
    
    size_t vl;
    for (size_t i = 0; i < nb_elements; i += vl) {
        // 1. 设置向量长度（基于uint64元素）
        vl = __riscv_vsetvl_e64m4(nb_elements - i);
        
        // 2. 加载uint64向量
        vuint64m4_t v_u64 = __riscv_vle64_v_u64m4(input_buffer + i, vl);
        
        // 3. 减去零点（zero_point） - 无符号减法
        vuint64m4_t v_offset = __riscv_vsub_vx_u64m4(v_u64, zero_point, vl);
        
        // 4. 将uint64转换为double（保持精度）
        vfloat64m4_t v_f64 = __riscv_vfcvt_f_xu_v_f64m4_rm(v_offset, __RISCV_FRM_RMM, vl);
        
        // 5. 应用缩放（使用double精度）
        vfloat64m4_t v_scaled = __riscv_vfmul_vf_f64m4(v_f64, scale_double, vl);
        
        // 6. 将double转换为float32（窄化转换）
        vfloat32m2_t v_f32 = __riscv_vfncvt_f_f_w_f32m2_rm(v_scaled, __RISCV_FRM_RMM, vl);
        
        // 7. 存储float32结果
        __riscv_vse32_v_f32m2(output_buffer + i, v_f32, vl);
    }
    return 0;
}

RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float32_to_float16, 22.11)
int rte_ml_io_float32_to_float16(const void *input, void *output, uint64_t nb_elements)
{
    // 参数检查
    if ((nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;

    const float *input_buffer = (const float *)input;
    _Float16 *output_buffer = (_Float16 *)output;

    size_t vl;
    for (size_t i = 0; i < nb_elements; i += vl) {
        // 1. 设置向量长度（基于float32元素）
        // 使用m4向量寄存器组处理更多元素
        vl = __riscv_vsetvl_e32m4(nb_elements - i);
        
        // 2. 加载float32向量
        vfloat32m4_t v_f32 = __riscv_vle32_v_f32m4(input_buffer + i, vl);
        
        // 3. 将float32转换为float16（使用最近偶数舍入模式）
        vfloat16m2_t v_f16 = __riscv_vfncvt_f_f_w_f16m2(v_f32, vl);
        
        // 4. 存储float16结果（直接存储到uint16_t缓冲区）
        __riscv_vse16_v_f16m2((_Float16 *)(output_buffer + i), v_f16, vl);
    }
    return 0;
}


RTE_EXPORT_EXPERIMENTAL_SYMBOL(rte_ml_io_float16_to_float32, 22.11)
int rte_ml_io_float16_to_float32(const void *input, void *output, uint64_t nb_elements)
{
    // 参数检查
    if ((nb_elements == 0) || (input == NULL) || (output == NULL))
        return -EINVAL;

    const _Float16 *input_buffer = (const _Float16 *)input;
    float *output_buffer = (float *)output;

    size_t vl;
    for (size_t i = 0; i < nb_elements; i += vl) {
        // 1. 设置向量长度（基于float16元素）
        // 使用m4向量寄存器组处理更多元素
        vl = __riscv_vsetvl_e16m4(nb_elements - i);
        
        // 2. 加载float16向量（直接加载uint16_t数据）
        vfloat16m4_t v_f16 = __riscv_vle16_v_f16m4(
            (const _Float16*)(input_buffer + i), vl);
        
        // 3. 将float16转换为float32（扩展转换）
        // 使用宽化转换指令，元素数量不变但位宽加倍
        vfloat32m8_t v_f32 = __riscv_vfwcvt_f_f_v_f32m8(v_f16, vl);
        
        // 4. 存储float32结果
        __riscv_vse32_v_f32m8(output_buffer + i, v_f32, vl);
    }
    return 0;
}




