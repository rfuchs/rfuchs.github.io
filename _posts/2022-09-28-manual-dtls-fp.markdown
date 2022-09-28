---
layout: post
title:  "Manually determine DTLS certificate fingerprint from a pcap"
---
RTP clients using DTLS-SRTP in combination with an SDP exchange use a fingerprint
mechanism to verify that the DTLS handshake is in fact happening with the correct
peer, and not some third-party MITM.

The fingerprint is derived from some digest hash function (generally SHA-256,
or SHA-1 for older clients), usually by calling some library function such as
`X509_digest()` on the object representing the certificate. The resulting digest
is then insert into the SDP:

```
a=fingerprint:sha-1 E0:16:DB:2C:84:34:D0:AC:77:CA:1F:80:D4:0D:36:34:09:AF:AA:8C
```

For debugging purposes I started wondering if it's possible to determine this
digest from the raw DTLS exchange as it would be captured by Wireshark or tcpdump.
Turns out it is.

In Wireshark, look for the DTLS packet containing the certificate. It should be
clearly marked with `Certificate`. Expanding the view should also reveal the details
of the contents of the certificate.

![Wireshark showing the certificate](/assets/wireshark-dtls-cert.png)

Select the single certificate. Right-click and use of the copy or export options.
Probably the easiest way is to use "export packet bytes" and save the certificate
to a file.

![Wireshark export bytes](/assets/wireshark-export-bytes.png)

The resulting file should match the certificate length given by Wireshark. It should
also be recognised by `file` as a certificate if `file` is up-to-date enough.

![Wireshark certificate length](/assets/wireshark-cert-length.png)

And this is the certificate that needs to go through the hash function to produce
the digest that is presented in the SDP.

![Final digest](/assets/sha1sum.png)
