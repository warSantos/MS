def get_docs_chi2(one_hot, chi2_values):

    chi2_docs_values = []
    # Para cada representação one-hot de cada documento.
    for doc in one_hot.toarray():
        doc_score = 0
        words_indexes = np.where(doc > 0)[0]
        for index in words_indexes:
            #doc_score += chi2_values[index] * doc[index]
            doc_score += chi2_values[index]
        # Verifica se o documento teve ao menos uma palavra.
        if words_indexes.shape[0] > 0:
            # Armazena o valor de chi2 do documento normalizado pelo número de palavras.
            chi2_docs_values.append(doc_score / words_indexes.shape[0])
        else:
            chi2_docs_values.append(0)
    return chi2_docs_values
