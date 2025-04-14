'''
returns how much a document moved up, measured by reciprocal ranking
'''

class RetrievalMetric:
    def __init__(self, original_ranking, rewritten_ranking):
        """
        Initialize the class with the original and rewritten rankings.
        
        original_ranking: List of document IDs in the original ranked order.
        rewritten_ranking: List of document IDs in the rewritten ranked order.
        """
        self.original_ranking = original_ranking
        self.rewritten_ranking = rewritten_ranking

    def reciprocal_rank(self, rank):
        """Calculate reciprocal rank for a given position (1-based index)."""
        return 1 / rank if rank > 0 else 0

    def find_rank(self, document_id, ranking_list):
        """Find the rank of a document in the given ranking list (1-based index)."""
        if document_id in ranking_list:
            return ranking_list.index(document_id) + 1  # +1 to convert to 1-based index
        else:
            return -1  # Return -1 if the document is not found in the list

    def measure_movement(self, target_document):
        """
        Measure the movement of the target document using MRR.
        
        :param target_document: The ID of the target document to measure movement for.
        :return: A dictionary containing the movement details.
        """
        # Find the rank of the target document in both rankings
        original_rank = self.find_rank(target_document, self.original_ranking)
        rewritten_rank = self.find_rank(target_document, self.rewritten_ranking)
        
        if original_rank == -1 or rewritten_rank == -1:
            return "Target document not found in one or both rankings."
        
        # Calculate reciprocal ranks before and after
        original_rr = self.reciprocal_rank(original_rank)
        rewritten_rr = self.reciprocal_rank(rewritten_rank)
        
        # Calculate how much the target document has moved upwards in terms of MRR
        movement = original_rr - rewritten_rr
        
        return {
            "original_rank": original_rank,
            "rewritten_rank": rewritten_rank,
            "original_rr": original_rr,
            "rewritten_rr": rewritten_rr,
            "movement": movement
        }

# Example usage:
original_ranking = [1, 2, 3, 4, 5, 6]  # Example original ranking
rewritten_ranking = [3, 1, 2, 4, 5, 6]  # Example rewritten ranking
target_document = 3  # Target document to measure

ranking_movement = RetrievalMetric(original_ranking, rewritten_ranking)
result = ranking_movement.measure_movement(target_document)
print(result['movement'])
