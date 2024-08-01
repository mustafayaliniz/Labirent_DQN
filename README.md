# Labirent DQN

Labirent DQN, bir derin Q-öğrenme (DQN) algoritması kullanarak bir labirent ortamında bir oyuncunun (maymun) muzları hedef olarak belirlemesini sağlayan bir yapay zeka projesidir. Proje, pygame kütüphanesi ile bir labirent oyunu ve PyTorch ile bir DQN modelini içerir.

## İçindekiler

- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
  - [DQN Modeli](#dqn-modeli)
  - [Kod Yapısı](#kod-yapısı)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [İletişim](#iletişim)

## Özellikler

- **Labirent**: Dinamik olarak oluşturulmuş bir labirent yapısı. Duvarlar ve yolları temsil eden hücrelerden oluşur.
- **Görseller**: Oyuncu (maymun) ve muz için bitmap görselleri.
- **DQN Ajanı**: Derin Q-öğrenme algoritması kullanılarak eğitim gerçekleştirilir. Bu ajan, çeşitli eylem ve durum kombinasyonlarına tepki verir.
- **Epsilon-Greedy Stratejisi**: Eylem seçiminde keşif ve sömürü arasındaki dengeyi sağlamak için epsilon-greedy stratejisi kullanılır.
- **Replay Buffer**: Deneyimlerin saklandığı ve tekrar oynatıldığı bir buffer yapısı.
- **Hedef Model Güncelleme**: Modelin öğrenme sürecinde istikrarı artırmak için hedef model belirli aralıklarla güncellenir.

## Gereksinimler

- **Python 3.x**: Projenin çalışması için Python 3.x sürümü gereklidir.
- **Pygame**: Oyun geliştirme ve grafik işleme kütüphanesi.
- **NumPy**: Sayısal işlemler ve matris manipülasyonu için.
- **PyTorch**: Derin öğrenme ve model eğitimi için.

### Kurulum

Projenin çalışması için gerekli olan kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install pygame numpy torch
```

## Kullanım

### DQN Modeli

Proje, derin Q-öğrenme algoritması için iki ana bileşen içerir:

**`DQNModel`**: PyTorch kullanılarak oluşturulmuş derin öğrenme modelidir. Model üç tam bağlantılı (fully connected) katmandan oluşur:
- **`fc1`**: İlk tam bağlantılı katman (24 nöron).
- **`fc2`**: İkinci tam bağlantılı katman (24 nöron).
- **`fc3`**: Çıktı katmanı, eylem sayısına göre boyutlandırılmıştır.

Model, `relu` aktivasyon fonksiyonları kullanır ve çıktı olarak Q-değerlerini tahmin eder.

**`DQNAgent`**: Bu sınıf, `DQNModel`'i yönetir ve eğitim sürecini yürütür:
- **`predict(state)`**: Verilen bir durumdan Q-değerlerini tahmin eder.
- **`fit(state, target)`**: Modeli verilen durum ve hedef değerler ile eğitir. Hedef Q-değerleri, Bellman denklemi kullanılarak hesaplanır.

### Kod Yapısı

- **`__init__`**: Pygame başlatma, labirent ve görsel yükleme, DQN ajanlarının oluşturulması ve ayarların yapılması.
- **`draw_maze`**: Labirenti ekrana çizer. Labirent hücrelerinin renkleri, duvarlar ve yolları ayırt eder.
- **`draw_monkey`**: Oyuncu (maymun) görselini ekrana çizer.
- **`draw_banana`**: Muz görselini ekrana çizer.
- **`move_player(action)`**: Oyuncunun belirtilen eyleme göre hareketini yönetir. Oyuncu müze ulaştığında ödül hesaplanır ve labirent sıfırlanır.
- **`get_state(pos)`**: Oyuncunun pozisyonunu bir durum vektörüne dönüştürür. Bu vektör, modelin girdi olarak kullanacağı bir formattadır.
- **`choose_action(state)`**: Epsilon-greedy stratejisi ile eylem seçimi yapar. Epsilon değeri keşif ve sömürü arasındaki dengeyi sağlar.
- **`update_target_model()`**: Hedef modelin ağırlıklarını günceller. Bu işlem, hedef modelin istikrarını sağlamak için yapılır.
- **`store_transition(state, action, reward, next_state)`**: Geçişleri replay buffer'a ekler. Bu buffer, ajanı deneyimlerden öğrenmek için kullanılır.
- **`replay_experience()`**: Replay buffer'daki deneyimleri tekrar oynatarak modelin öğrenmesini sağlar. Eğitim verisi küçükse, bu işlem yapılmaz.
- **`run()`**: Ana döngü; oyun akışını kontrol eder. Oyun ekranını günceller, eylemleri seçer, oyuncuyu hareket ettirir ve zamanlayıcıyı yönetir.

## Katkıda Bulunma

Katkılarınızı memnuniyetle kabul ediyoruz! Lütfen katkıda bulunmadan önce [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakabilirsiniz.

## İletişim

Proje ile ilgili sorularınız için [uzayk204@gmail.com](mailto:uzayk204@gmail.com) adresinden iletişime geçebilirsiniz.
