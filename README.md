# Phone-filter
## Datas
```c
price: ...      (fiyat için)
brand: ...      (marka için  samsung/apple/xiaomi/huawei )
os: ...         (işletim sistemi için android/ios )
usage: ...      (ne için kullanılcağı gaming/photo/none )
ram: ...        (ram kapasitesi için  4 6 8 12 16 )
storage: ...    (hafıza kapasitesi için good/bad veya 128 256 )
battery: ...    (batarya kapasitesi için good/bad )
camera: ...     (camera için good/bad )
screen: ...     (ekran için)
```


## Dataset data type

veriler şu şekilde girilmeli:<br> `input -> output` <br><br>
`input -> price:...; brand:...; os:...; usage:...; ram:...; storage:...; battery:...; camera:...; screen:...; ` <br><br>
`blablablabla -> price:...; brand:...; os:...; usage:...; ram:...; storage:...; battery:...; camera:...; screen:...; `
### örnek

```json
fiyatı 15000 tl olan oyun oynamak için 6 gb ramli android xiaomi iyi bataryalı ve hafızalı telefon -> price: 15000; brand: xiaomi; os:android; usage:game; ram: 6; storage:good; battery:good; camera:none; screen:none;
```

```json
  price: 15000; brand: xiaomi; os:android; usage:game; ram: 6; storage:good; battery:good; camera:none; screen:none;
```


### başka örnekler
```json
fiyatı 12000 tl olan android telefon -> price: 12000; brand: none; os:android; usage:none; ram: none; storage:none; battery:none; camera:none; screen:none;
selfie için iphone -> price: any; brand: apple; os:ios; usage:photo; ram: none; storage:none; battery:none; camera:good; screen:none;
```
### dataset linki:

https://github.com/zyr1on/Phone-filter/blob/main/t5/training_data.txt
