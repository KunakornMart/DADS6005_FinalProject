# DADS6005 Final Project (DADS5)

## 🧑‍🤝‍🧑 Members
| ลำดับ | ชื่อสมาชิก                   | รหัสนักศึกษา   |
|-------|----------------------------|----------------|
| 1     | นายฮาฟิซ เบ็ญราฮีม        | 6520422005     |
| 2     | นายรวีนท์ สมิทธิเมธินทร์   | 6610422008     |
| 3     | นางสาวธัชพรรณ สัมพันธ์สมโภช | 6610422012   |
| 4     | นายกฤชกร นิธิโชติภาคิน     | 6610422015     |
| 5     | นายนิวัฒน์ วุฒิศรีศิริพร   | 6610422016     |
| 6     | นายคุณากร พฤกษากร         | 6610422020     |
| 7     | นายณัฐวุฒิ อินต๊ะนัย       | 6610422023     |
| 8     | นายกฤษฎา อรัญชราธร        | 6610422026     |

---
# การพยากรณ์ราคา BTC ด้วยโมเดลต่าง ๆ
![Overview](https://drive.google.com/uc?id=1f9ldfmz2gUWrgVchHNrQ8q9-hw-44iEN)

## ผลลัพธ์จากการรันโมเดล

1. **Linear Regression**  
   ![Linear Regression Result](https://drive.google.com/uc?id=1WaCaEiWSx5lahe--ZyHhfEW_5bw8VsHe)

2. **Prophet Model**  
   ![Decision Tree Result](https://drive.google.com/uc?id=1kLjx5wbf_SUw4tbYBS8OBgWtTh4c3QCY)

3. **XGBoost**  
   ![XGBoost Result](https://drive.google.com/uc?id=19qIAhTvp-4uQpVqjRoJsURy0oZP5EtQ7)

4. **Random Forest**  
   ![Random Forest Result](https://drive.google.com/uc?id=1AeYVEQ01OekuTwwJMK6gCRJz5mgrxuI6)

5. **LSTM**  
   ![LSTM Result](https://drive.google.com/uc?id=1ZlicYoEMlmfCIOELqmDPbtKtgU3L2NB2)


## 📈 ML Models Selection
ทดลองรัน ML Model จำนวน 5 models ในวันที่ 20 ธันวาคม 2567 เพื่อพิจารณาเลือก Model ที่เหมาะสมสำหรับการทำนายราคา BTC

### ผลลัพธ์ค่า MAPE ของทั้ง 5 models:
| ลำดับ | ML Model             | เฉลี่ยค่า MAPE       |
|-------|----------------------|--------------------|
| 1     | Linear Regression    | 0.344504473       |
| 2     | Prophet ✅            | 0.290467437       |
| 3     | XGBoost              | 0.310918966       |
| 4     | RandomForest         | 0.425594191       |
| 5     | LSTM                 | 477,512,879.55    |

---

## 🌟 เหตุผลที่เลือก Prophet Model

Prophet เป็นโมเดลที่พัฒนาโดยทีมงาน Facebook (Meta) สำหรับการพยากรณ์ข้อมูลแบบ time series โดยเหมาะสำหรับข้อมูลที่มีลักษณะเป็นแนวโน้ม (trend) และฤดูกาล (seasonality) Prophet สามารถรับมือกับข้อมูลที่ผันผวนและ missing data ได้ดี และเหมาะสำหรับการวิเคราะห์ข้อมูลราคาของ BTC ซึ่งมีความผันผวนสูง

เหตุผลที่ Prophet เป็นโมเดลที่เหมาะสม เพราะข้อมูลราคาของ BTC มีลักษณะเป็นข้อมูลแบบ time series และ  Prophet ถูกออกแบบมาโดยเฉพาะเพื่อจัดการข้อมูลประเภทนี้ ไม่ว่าจะเป็นข้อมูลที่มี trend แนวโน้มเพิ่มขึ้น ลดลง หรือมีความผันผวนสูง ซึ่ง Cryptocurrency มีความผันผวนสูงอยู่แล้ว และ Prophet สามารถรับมือกับ missing data ได้ดี เนื่องจากมีการปรับแต่งพารามิเตอร์โดยอัตโนมัติ ทำให้ลดความซับซ้อนในการเตรียมข้อมูลก่อนใช้งาน รวมทั้ง Prophet มีความสามารถพิเศษในการตรวจจับ seasonality เช่น แนวโน้มราคาที่เปลี่ยนแปลงตามช่วงเวลา รวมถึงผลกระทบจากวันสำคัญที่อาจส่งผลต่อราคา เช่น ราคา bitcoin ชอบปรับตัวลดลงในช่วงคริสต์มาส

---

## 🔮 การทำงานของ Prophet Model
1. **ดึงข้อมูล**: ดึงข้อมูลราคาของ BTC จาก MongoDB และจัดให้อยู่ในรูปแบบ DataFrame ที่มีคอลัมน์ `ds` (datetime) และ `y` (price)
2. **เทรนโมเดล**: ใช้คำสั่ง `model.fit(df)` เทรนโมเดลโดยตั้งค่าพารามิเตอร์ เช่น การตรวจจับ trend และ seasonality
3. **บันทึกโมเดล**: บันทึกโมเดลที่เทรนเสร็จแล้วลงในไฟล์ `model_prophet.pkl` ด้วย `pickle`
4. **พยากรณ์**: โหลดโมเดลที่บันทึกไว้ และสร้าง DataFrame ของช่วงเวลาที่ต้องการพยากรณ์ พร้อมพยากรณ์ด้วยคำสั่ง `predict`
5. **ประเมินผล**: คำนวณค่า MAPE โดยเปรียบเทียบค่าที่พยากรณ์กับราคาจริงทุก 12 รอบ (1 ชั่วโมง)

---
# รันโค้ดทั้งหมดด้วย Apache Airflow บน Amazon EC2 โดยใช้ Prophet Model ในการทำนายราคา Bitcoin (BTC)
![EC2](https://drive.google.com/uc?id=1dMTXIcNaa914zBxZm_qTu7C-je5MIzmd)

## 📊 ผลลัพธ์ค่า MAPE จาก Prophet Model

**ช่วงเวลา: 22.00น. ของวันที่ 21 ธันวาคม 2567 - 6.00น. ของวันที่ 22 ธันวาคม 2567**  
   ![21/12/67](https://drive.google.com/uc?id=1y-LmiSZlzC6S0aPdDUq3r_kTQRXAV0jh)

**ช่วงเวลา: 22.00น. ของวันที่ 22 ธันวาคม 2567 - 6.00น. ของวันที่ 23 ธันวาคม 2567**  
   ![22/12/67](https://drive.google.com/uc?id=1up1v7taT9rSKYBhyaI9GpyfLnOsaMrp8)

**ช่วงเวลา: 22.00น. ของวันที่ 23 ธันวาคม 2567 - 6.00น. ของวันที่ 24 ธันวาคม 2567**  
   ![23/12/67](https://drive.google.com/uc?id=19RU94p3jKaBHfF9Agtq_dBbwxN9sNF-s)
   
### ผลลัพธ์ค่า MAPE ของทั้ง 3 วัน:
| ช่วงเวลา                                      | ค่าเฉลี่ย MAPE |
|--------------------------------------------|---------------|
| 22.00น. ของวันที่ 21 - 6.00น. ของวันที่ 22 ธ.ค. 2567 | 0.275053654           |
| 22.00น. ของวันที่ 22 - 6.00น. ของวันที่ 23 ธ.ค. 2567 | 0.423694658          |
| 22.00น. ของวันที่ 23 - 6.00น. ของวันที่ 24 ธ.ค. 2567 | 0.387907306          |

**🔗 ดูโค้ดของทุกโมเดลและผลลัพธ์เพิ่มเติมได้ที่:** [Google Drive](https://drive.google.com/drive/folders/1bSWRsJju8P64TNbWimIEthTEEHGv72Mj?usp=sharing )
