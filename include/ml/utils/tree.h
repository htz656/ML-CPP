#ifndef TREE_H
#define TREE_H

#include <vector>
#include <functional>

template <typename T>
class Tree {
private:
    struct TreeNode {
        T data;
        TreeNode* parent;
        std::vector<TreeNode*> children;

        TreeNode(const T& value) : data(value), parent(nullptr) {}
    };

    TreeNode* root;

    void destroySubtree(TreeNode* node) {
        if (!node) return;
        for (auto& child : node->children) {
            destroySubtree(child);
        }
        delete node;
    }

public:
    Tree() : root(nullptr) {}
    ~Tree() { destroySubtree(root); }

    TreeNode* createRoot(const T& value) {
        if (root) {
            destroySubtree(root);
        }
        root = new TreeNode(value);
        return root;
    }

    TreeNode* addChild(TreeNode* parent, const T& value) {
        if (!parent) return nullptr;
        TreeNode* newNode = new TreeNode(value);
        newNode->parent = parent;
        parent->children.push_back(newNode);
        return newNode;
    }

    void traversePreOrder(const TreeNode* node, 
                        std::function<void(const T&)> callback) const {
        if (!node) return;
        callback(node->data);
        for (const auto& child : node->children) {
            traversePreOrder(child, callback);
        }
    }

    void traversePostOrder(const TreeNode* node,
                         std::function<void(const T&)> callback) const {
        if (!node) return;
        for (const auto& child : node->children) {
            traversePostOrder(child, callback);
        }
        callback(node->data);
    }

    TreeNode* getRoot() const { return root; }

    TreeNode* findNode(const TreeNode* startNode, const T& value) const {
        if (!startNode) return nullptr;
        if (startNode->data == value) return const_cast<TreeNode*>(startNode);
        
        for (const auto& child : startNode->children) {
            TreeNode* found = findNode(child, value);
            if (found) return found;
        }
        return nullptr;
    }
};

#endif 