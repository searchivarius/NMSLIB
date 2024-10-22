#include "method/hnsw_distfunc_mv.h"

#ifdef EMAX7
#define NULL ((void *)0)
#define IMAX_KERNEL_COL_SIZE 60
#define NCHIP 1

int imax_search_mv(float curdist, int curNodeNum, float *pVectq, int *data, size_t qty, size_t size, char *data_level0_memory_, size_t memoryPerObject_, size_t offsetData_) {
    #if defined(ARMZYNQ)
    if (membase == NULL) {
        sysinit(1024*1024*1024, 32);
    }
    #endif
    int imax_emb = qty % IMAX_KERNEL_COL_SIZE ? ((qty/IMAX_KERNEL_COL_SIZE) + 1)*IMAX_KERNEL_COL_SIZE : qty;
    int imax_size = size % 8 ? ((size/8) + 1)*8 : size;
    #ifdef ARMZYNQ
    float *imax_key_array = (float *)imax_alloc(imax_size*imax_emb*sizeof(float), 32);
    float *imax_query_array = (float *)imax_alloc(imax_emb*sizeof(float), 32);
    float *imax_result_array = (float *)imax_alloc(imax_size*sizeof(float), 32);
    #else
    float imax_key_array[imax_size*imax_emb];
    float imax_query_array[imax_emb];
    float imax_result_array[imax_size];
    #endif
    bool changed = false;
    for (int j = 1; j <= imax_size; j++) {
        int tnum = *(data + j);
        for (int k = 0; k < imax_emb; k++) {
            if (k < qty) {
                imax_key_array[(j-1)*imax_size + k] = *(data_level0_memory_ + tnum * memoryPerObject_ + offsetData_ + 16 + k);
                imax_query_array[k] = pVectq[k];
            } else {
                imax_key_array[(j-1)*imax_size + k] = 0;
                imax_query_array[k] = 0;
            }
        }
    }

    for (int emb_blk_idx = 0; emb_blk_idx < imax_emb / IMAX_KERNEL_COL_SIZE; emb_blk_idx++) {
        Ull qaddr = (Ull)&imax_query_array[emb_blk_idx*IMAX_KERNEL_COL_SIZE];
        Ull kaddr[IMAX_KERNEL_COL_SIZE * 4];
        for (int j = 0; j < IMAX_KERNEL_COL_SIZE; j++) {
            for (int k = 0; k < 4; k++) {
                kaddr[(j*4) + k] = (Ull)&imax_key_array[(emb_blk_idx*IMAX_KERNEL_COL_SIZE*imax_size) + (j*imax_size) + k*2];
            }
        }

        Ull raddr[4];
        for (int j = 0; j < 4; j++) {
            raddr[j] = (Ull)&imax_result_array[j*2];
        }

        Ull CHIP, LOLP, INIT0, INIT1, LOOP0, LOOP1;
        Ull cofs, rofs;
        Ull fetch_size = imax_size * 4;
        Ull rofs_init = 0<<32|(0-4*8LL)&0xFFFFFFFF;
        Ull BR[64][4][4], AR[64][4];

#define mv1_core(r, rm1, k0, k1, k2, k3, q) \
                mop(OP_LDR, 3, &BR[rm1][0][1], (Ull)kaddr[k0], (Ull)rofs, MSK_W0, (Ull)kaddr[k0], fetch_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_LDR, 3, &BR[rm1][0][0], (Ull)kaddr[k1], (Ull)rofs, MSK_W0, (Ull)kaddr[k0], fetch_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_LDR, 3, &BR[rm1][1][1], (Ull)kaddr[k2], (Ull)rofs, MSK_W0, (Ull)kaddr[k0], fetch_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_LDR, 3, &BR[rm1][1][0], (Ull)kaddr[k3], (Ull)rofs, MSK_W0, (Ull)kaddr[k0], fetch_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_LDR, 3, &BR[rm1][2][1], (Ull)qaddr, q*8LL, MSK_W0, (Ull)qaddr, imax_emb, 0, 0, (Ull)NULL, 0); \
                exe(OP_FMA, &AR[r][3], BR[rm1][0][1], EXP_H3210, BR[rm1][2][1], EXP_H3210, AR[rm1][3], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                exe(OP_FMA, &AR[r][2], BR[rm1][0][0], EXP_H3210, BR[rm1][2][1], EXP_H3210, AR[rm1][2], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                exe(OP_FMA, &AR[r][1], BR[rm1][1][1], EXP_H3210, BR[rm1][2][1], EXP_H3210, AR[rm1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                exe(OP_FMA, &AR[r][0], BR[rm1][1][0], EXP_H3210, BR[rm1][2][1], EXP_H3210, AR[rm1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define mv1_store(r, rm1) \
                mop(OP_LDR, 3, &BR[rm1][0][1], (Ull)raddr[0], (Ull)rofs, MSK_W0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_LDR, 3, &BR[rm1][0][0], (Ull)raddr[1], (Ull)rofs, MSK_W0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_LDR, 3, &BR[rm1][1][1], (Ull)raddr[2], (Ull)rofs, MSK_W0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_LDR, 3, &BR[rm1][1][0], (Ull)raddr[3], (Ull)rofs, MSK_W0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0); \
                exe(OP_FMA, &AR[r][3], BR[rm1][0][1], EXP_H3210, AR[rm1][3], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                exe(OP_FMA, &AR[r][2], BR[rm1][0][0], EXP_H3210, AR[rm1][2], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                exe(OP_FMA, &AR[r][1], BR[rm1][1][1], EXP_H3210, AR[rm1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                exe(OP_FMA, &AR[r][0], BR[rm1][1][0], EXP_H3210, AR[rm1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                mop(OP_STR, 3, &AR[r][3], (Ull)raddr[0], (Ull)rofs, MSK_D0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_STR, 3, &AR[r][2], (Ull)raddr[1], (Ull)rofs, MSK_D0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_STR, 3, &AR[r][1], (Ull)raddr[2], (Ull)rofs, MSK_D0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0); \
                mop(OP_STR, 3, &AR[r][0], (Ull)raddr[3], (Ull)rofs, MSK_D0, (Ull)raddr[0], imax_size, 0, 0, (Ull)NULL, 0)
//EMAX5A begin mv1 mapdist=0
        for (CHIP=0;CHIP<NCHIP;CHIP++) {
            for (INIT1=1,LOOP1=1,cofs=0;LOOP1--;INIT1=0) {
                for (INIT0=1,LOOP0=imax_size/8,rofs=rofs_init;LOOP0--;INIT0=0) {
                    exe(OP_ADD, &rofs, INIT0?rofs:rofs, EXP_H3210, 0<<32|(4*8LL)&0xffffffff, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);

                    mop(OP_LDR, 3, &BR[1][0][1], (Ull)kaddr[0], (Ull)rofs, MSK_W0, (Ull)kaddr[0], fetch_size, 0, 0, (Ull)NULL, fetch_size);
                    mop(OP_LDR, 3, &BR[1][0][0], (Ull)kaddr[1], (Ull)rofs, MSK_W0, (Ull)kaddr[0], fetch_size, 0, 0, (Ull)NULL, fetch_size);
                    mop(OP_LDR, 3, &BR[1][1][1], (Ull)kaddr[2], (Ull)rofs, MSK_W0, (Ull)kaddr[0], fetch_size, 0, 0, (Ull)NULL, fetch_size);
                    mop(OP_LDR, 3, &BR[1][1][0], (Ull)kaddr[3], (Ull)rofs, MSK_W0, (Ull)kaddr[0], fetch_size, 0, 0, (Ull)NULL, fetch_size);
                    mop(OP_LDR, 3, &BR[1][2][1], (Ull)qaddr,  (Ull)0LL,  MSK_W0, (Ull)qaddr,  imax_emb,   0, 0, (Ull)NULL, imax_emb);

                    exe(OP_FML, &AR[2][3], BR[1][0][1], EXP_H3210, BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                    exe(OP_FML, &AR[2][2], BR[1][0][0], EXP_H3210, BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                    exe(OP_FML, &AR[2][1], BR[1][0][1], EXP_H3210, BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                    exe(OP_FML, &AR[2][0], BR[1][0][0], EXP_H3210, BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

                    mv1_core( 3,  2,  4,  5,  6,  7,  1);mv1_core( 4,  3,  8,  9, 11, 12,  2);
                    mv1_core( 5,  4, 13, 14, 15, 16,  3);mv1_core( 6,  5,  4, 17, 18, 19,  4);
                    mv1_core( 7,  6, 20, 21, 22, 23,  5);mv1_core( 8,  7, 24, 25, 26, 27,  6);
                    mv1_core( 9,  8, 28, 29, 30, 31,  7);mv1_core(10,  9, 32, 33, 34, 35,  8);
                    mv1_core(11, 10, 36, 37, 38, 39,  9);mv1_core(12, 11, 40, 41, 42, 43, 10);
                    mv1_core(13, 12, 44, 45, 46, 47, 11);mv1_core(14, 13, 48, 49, 50, 51, 12);
                    mv1_core(15, 14, 52, 53, 54, 55, 13);mv1_core(16, 15, 56, 57, 58, 59, 14);
                    mv1_core(17, 16, 60, 61, 62, 63, 15);mv1_core(18, 17, 64, 65, 66, 67, 16);
                    mv1_core(19, 18, 68, 69, 70, 71, 17);mv1_core(20, 19, 72, 73, 74, 75, 18);
                    mv1_core(21, 20, 76, 77, 78, 79, 19);mv1_core(22, 21, 80, 81, 82, 83, 20);
                    mv1_core(23, 22, 84, 85, 86, 87, 21);mv1_core(24, 23, 88, 89, 90, 91, 22);
                    mv1_core(25, 24, 92, 93, 94, 95, 23);mv1_core(26, 25, 96, 97, 98, 99, 24);
                    mv1_core(27, 26,100,101,102,103, 25);mv1_core(28, 27,104,105,106,107, 26);
                    mv1_core(29, 28,108,109,110,111, 27);mv1_core(30, 29,112,113,114,115, 28);
                    mv1_core(31, 30,116,117,118,119, 29);mv1_core(32, 31,120,121,122,123, 30);
                    mv1_core(33, 32,124,125,126,127, 31);mv1_core(34, 33,128,129,130,131, 32);
                    mv1_core(35, 34,132,133,134,135, 33);mv1_core(36, 35,136,137,138,139, 34);
                    mv1_core(37, 36,140,141,142,143, 35);mv1_core(38, 37,144,145,146,147, 36);
                    mv1_core(39, 38,148,149,150,151, 37);mv1_core(40, 39,152,153,154,155, 38);
                    mv1_core(41, 40,156,157,158,159, 39);mv1_core(42, 41,160,161,162,163, 40);
                    mv1_core(43, 42,164,165,166,167, 41);mv1_core(44, 43,168,169,170,171, 42);
                    mv1_core(45, 44,172,173,174,175, 43);mv1_core(46, 45,176,177,178,179, 44);
                    mv1_core(47, 46,180,181,182,183, 45);mv1_core(48, 47,184,185,186,187, 46);
                    mv1_core(49, 48,188,189,190,191, 47);mv1_core(50, 49,192,193,194,195, 48);
                    mv1_core(51, 50,196,197,198,199, 49);mv1_core(52, 51,200,201,202,203, 50);
                    mv1_core(53, 52,204,205,206,207, 51);mv1_core(54, 53,208,209,210,211, 52);
                    mv1_core(55, 54,212,213,214,215, 53);mv1_core(56, 55,216,217,218,219, 54);
                    mv1_core(57, 56,220,221,222,223, 55);mv1_core(58, 57,224,225,226,227, 56);
                    mv1_core(59, 58,228,229,230,231, 57);mv1_core(60, 59,232,233,234,235, 58);
                    mv1_core(61, 60,236,237,238,239, 59);

                    mv1_store(62, 61);
                }
            }
        }
//EMAX5A end

    }
//EMAX5A drain_dirty_lmm

    float minDist = INFINITY;
    for (int j = 1; j <= size; j++) {
        float result = -*(float*)&imax_result_array[j-1];
        if (result < minDist) {
            minDist = result;
        }
        if (minDist < curdist) {
            curdist = minDist;
            curNodeNum = *(data + j);
            changed = true;
        }
    }
    return changed;
}
#endif