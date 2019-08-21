import iisignature
import signatory

import utils_testing as utils


class TestLyndon(utils.TimedUnitTest):
    @classmethod
    def iisignature_convert(cls, ii_elem):
        outstr = ''
        for character in ii_elem:
            if character == '1':
                outstr += '0'
            elif character == '2':
                outstr += '1'
            elif character == '3':
                outstr += '2'
            elif character == '4':
                outstr += '3'
            elif character == '5':
                outstr += '4'
            elif character == '6':
                outstr += '5'
            elif character == '7':
                outstr += '6'
            elif character == '8':
                outstr += '7'
            elif character == '9':
                outstr += '8'
            elif character == '?':
                outstr += '9'
            else:
                outstr += character
        return outstr

    def test_brackets(self):
        for channels in range(2, 11):  # iisignature supports channels with unique symbols in the range 2 to 10
                                       # inclusive
            for depth in range(1, 6):
                iisignature_brackets = iisignature.basis(iisignature.prepare(channels, depth))
                signatory_brackets = signatory.lyndon_brackets(channels, depth)
                for ii_elem, sig_elem in zip(iisignature_brackets, signatory_brackets):
                    if sig_elem != eval(self.iisignature_convert(ii_elem)):
                        self.fail("\n"
                                  "channels={channels}\n"
                                  "depth={depth}\n"
                                  "ii_elem={ii_elem}\n"
                                  "sig_elem={sig_elem}"
                                  .format(channels=channels, depth=depth, ii_elem=ii_elem, sig_elem=sig_elem))

    def test_words(self):
        for channels in range(2, 11):  # iisignature supports channels with unique symbols in the range 2 to 10
                                       # inclusive
            for depth in range(1, 6):
                iisignature_brackets = iisignature.basis(iisignature.prepare(channels, depth))
                signatory_words = signatory.lyndon_words(channels, depth)
                for ii_elem, sig_elem in zip(iisignature_brackets, signatory_words):
                    ii_elem_new = ii_elem.replace('[', '').replace(']', '').replace(',', '')
                    ii_elem_new = self.iisignature_convert(ii_elem_new)
                    sig_elem_new = ''.join(str(i) for i in sig_elem)
                    if sig_elem_new != ii_elem_new:
                        self.fail("channels={channels}\n"
                                  "depth={depth}\n"
                                  "ii_elem={ii_elem}\n"
                                  "sig_elem={sig_elem}\n"
                                  "ii_elem_new={ii_elem_new}\n"
                                  "sig_elem_new={sig_elem_new}"
                                  .format(channels=channels, depth=depth, ii_elem=ii_elem, sig_elem=sig_elem,
                                          ii_elem_new=ii_elem_new, sig_elem_new=sig_elem_new))

    def test_amount(self):
        for channels in range(1, 10):
            for depth in range(1, 6):
                words = len(signatory.lyndon_words(channels, depth))
                brackets = len(signatory.lyndon_brackets(channels, depth))
                if words != brackets:
                    self.fail("channels={channels}\n"
                              "depth={depth}\n"
                              "words={words}\n"
                              "brackets={brackets}"
                              .format(channels=channels, depth=depth, words=words, brackets=brackets))


class TestChannels(utils.TimedUnitTest):
    def test_signature_channels(self):
        for channels in range(1, 16):
            for depth in range(1, 15):
                result = signatory.signature_channels(channels, depth)
                sum_over = sum(channels ** i for i in range(1, depth + 1))
                if result != sum_over:
                    self.fail("channels={channels}\n"
                              "depth={depth}\n"
                              "result={result}\n"
                              "sum_over={sum_over}"
                              .format(channels=channels, depth=depth, result=result, sum_over=sum_over))

    def test_logsignature_channels(self):
        for channels in range(1, 10):
            for depth in range(1, 6):
                result = signatory.logsignature_channels(channels, depth)
                from_words = len(signatory.lyndon_words(channels, depth))
                if result != from_words:
                    self.fail("channels={channels}\n"
                              "depth={depth}\n"
                              "result={result}\n"
                              "from_words={from_words}"
                              .format(channels=channels, depth=depth, result=result, from_words=from_words))