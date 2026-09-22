#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    NANO_NULL = 0,
    NANO_NUMBER = 1,
    NANO_BOOL = 2,
    NANO_TEXT = 3,
    NANO_FUNCTION = 4,
    NANO_LIST = 5,
    NANO_OBJECT = 6,
} NanoTag;

typedef struct NanoValue NanoValue;

struct NanoValue {
    NanoTag tag;
    union {
        double number;
        int boolean;
        char *text;
        void *function;
        struct {
            size_t len;
            size_t cap;
            NanoValue **items;
        } list;
        struct {
            size_t len;
            size_t cap;
            char **keys;
            NanoValue **values;
        } object;
    } as;
};

static void nano_fail(const char *message) {
    fprintf(stderr, "Nano native: %s\n", message);
    exit(1);
}

static NanoValue *nano_new(NanoTag tag) {
    NanoValue *value = (NanoValue *)calloc(1, sizeof(NanoValue));
    if (!value) {
        nano_fail("out of memory");
    }
    value->tag = tag;
    return value;
}

static char *nano_strdup_value(const char *text) {
    size_t len = strlen(text);
    char *copy = (char *)malloc(len + 1);
    if (!copy) {
        nano_fail("out of memory");
    }
    memcpy(copy, text, len + 1);
    return copy;
}

NanoValue *nano_box_null(void) {
    return nano_new(NANO_NULL);
}

NanoValue *nano_box_number(double value) {
    NanoValue *out = nano_new(NANO_NUMBER);
    out->as.number = value;
    return out;
}

NanoValue *nano_box_bool(double value) {
    NanoValue *out = nano_new(NANO_BOOL);
    out->as.boolean = value != 0.0;
    return out;
}

NanoValue *nano_box_text(const char *value) {
    NanoValue *out = nano_new(NANO_TEXT);
    out->as.text = nano_strdup_value(value ? value : "");
    return out;
}

NanoValue *nano_box_function(void *value) {
    NanoValue *out = nano_new(NANO_FUNCTION);
    out->as.function = value;
    return out;
}

NanoValue *nano_list_new(void) {
    return nano_new(NANO_LIST);
}

NanoValue *nano_object_new(void) {
    return nano_new(NANO_OBJECT);
}

static void nano_list_reserve(NanoValue *list, size_t need) {
    if (list->as.list.cap >= need) {
        return;
    }
    size_t cap = list->as.list.cap ? list->as.list.cap * 2 : 4;
    while (cap < need) {
        cap *= 2;
    }
    NanoValue **items = (NanoValue **)realloc(list->as.list.items, cap * sizeof(NanoValue *));
    if (!items) {
        nano_fail("out of memory");
    }
    list->as.list.items = items;
    list->as.list.cap = cap;
}

static void nano_object_reserve(NanoValue *object, size_t need) {
    if (object->as.object.cap >= need) {
        return;
    }
    size_t cap = object->as.object.cap ? object->as.object.cap * 2 : 4;
    while (cap < need) {
        cap *= 2;
    }
    char **keys = (char **)realloc(object->as.object.keys, cap * sizeof(char *));
    NanoValue **values = (NanoValue **)realloc(object->as.object.values, cap * sizeof(NanoValue *));
    if (!keys || !values) {
        nano_fail("out of memory");
    }
    object->as.object.keys = keys;
    object->as.object.values = values;
    object->as.object.cap = cap;
}

static NanoValue *nano_clone(const NanoValue *value);

void nano_list_push(NanoValue *list, NanoValue *value) {
    if (!list || list->tag != NANO_LIST) {
        nano_fail("list_push requires List");
    }
    nano_list_reserve(list, list->as.list.len + 1);
    list->as.list.items[list->as.list.len++] = nano_clone(value);
}

void nano_object_put(NanoValue *object, const char *key, NanoValue *value) {
    if (!object || object->tag != NANO_OBJECT) {
        nano_fail("object_put requires Object");
    }
    for (size_t i = 0; i < object->as.object.len; ++i) {
        if (strcmp(object->as.object.keys[i], key) == 0) {
            object->as.object.values[i] = nano_clone(value);
            return;
        }
    }
    nano_object_reserve(object, object->as.object.len + 1);
    object->as.object.keys[object->as.object.len] = nano_strdup_value(key);
    object->as.object.values[object->as.object.len] = nano_clone(value);
    object->as.object.len++;
}

static NanoValue *nano_clone(const NanoValue *value) {
    if (!value) {
        return nano_box_null();
    }

    NanoValue *out = nano_new(value->tag);
    switch (value->tag) {
        case NANO_NULL:
            break;
        case NANO_NUMBER:
            out->as.number = value->as.number;
            break;
        case NANO_BOOL:
            out->as.boolean = value->as.boolean;
            break;
        case NANO_TEXT:
            out->as.text = nano_strdup_value(value->as.text);
            break;
        case NANO_FUNCTION:
            out->as.function = value->as.function;
            break;
        case NANO_LIST:
            nano_list_reserve(out, value->as.list.len);
            for (size_t i = 0; i < value->as.list.len; ++i) {
                out->as.list.items[out->as.list.len++] = nano_clone(value->as.list.items[i]);
            }
            break;
        case NANO_OBJECT:
            nano_object_reserve(out, value->as.object.len);
            for (size_t i = 0; i < value->as.object.len; ++i) {
                out->as.object.keys[out->as.object.len] = nano_strdup_value(value->as.object.keys[i]);
                out->as.object.values[out->as.object.len] = nano_clone(value->as.object.values[i]);
                out->as.object.len++;
            }
            break;
    }
    return out;
}

static int nano_truthy(const NanoValue *value) {
    if (!value) return 0;
    switch (value->tag) {
        case NANO_NULL: return 0;
        case NANO_NUMBER: return value->as.number != 0.0;
        case NANO_BOOL: return value->as.boolean != 0;
        case NANO_TEXT: return value->as.text[0] != '\0';
        case NANO_FUNCTION: return value->as.function != NULL;
        case NANO_LIST: return value->as.list.len != 0;
        case NANO_OBJECT: return value->as.object.len != 0;
    }
    return 0;
}

int nano_any_truthy(NanoValue *value) {
    return nano_truthy(value);
}

static int nano_value_equal(const NanoValue *a, const NanoValue *b) {
    if (!a || !b) return a == b;
    if (a->tag != b->tag) return 0;

    switch (a->tag) {
        case NANO_NULL: return 1;
        case NANO_NUMBER: return a->as.number == b->as.number;
        case NANO_BOOL: return a->as.boolean == b->as.boolean;
        case NANO_TEXT: return strcmp(a->as.text, b->as.text) == 0;
        case NANO_FUNCTION: return a->as.function == b->as.function;
        case NANO_LIST:
            if (a->as.list.len != b->as.list.len) return 0;
            for (size_t i = 0; i < a->as.list.len; ++i) {
                if (!nano_value_equal(a->as.list.items[i], b->as.list.items[i])) return 0;
            }
            return 1;
        case NANO_OBJECT:
            if (a->as.object.len != b->as.object.len) return 0;
            for (size_t i = 0; i < a->as.object.len; ++i) {
                int found = 0;
                for (size_t j = 0; j < b->as.object.len; ++j) {
                    if (strcmp(a->as.object.keys[i], b->as.object.keys[j]) == 0) {
                        if (!nano_value_equal(a->as.object.values[i], b->as.object.values[j])) return 0;
                        found = 1;
                        break;
                    }
                }
                if (!found) return 0;
            }
            return 1;
    }
    return 0;
}

static int nano_utf8_count(const char *text) {
    int count = 0;
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        if ((*p & 0xC0) != 0x80) count++;
        p++;
    }
    return count;
}

double nano_any_len(NanoValue *value) {
    if (!value) nano_fail("len() received null pointer");
    switch (value->tag) {
        case NANO_TEXT: return (double)nano_utf8_count(value->as.text);
        case NANO_LIST: return (double)value->as.list.len;
        case NANO_OBJECT: return (double)value->as.object.len;
        default: nano_fail("len() requires Text, List or Object");
    }
    return 0.0;
}

NanoValue *nano_any_index(NanoValue *target, NanoValue *index) {
    if (!target || !index) nano_fail("index received null value");

    if (target->tag == NANO_LIST && index->tag == NANO_NUMBER) {
        double n = index->as.number;
        if (n < 0.0 || floor(n) != n || n >= (double)target->as.list.len) {
            nano_fail("index out of bounds");
        }
        return nano_clone(target->as.list.items[(size_t)n]);
    }

    if (target->tag == NANO_OBJECT && index->tag == NANO_TEXT) {
        for (size_t i = 0; i < target->as.object.len; ++i) {
            if (strcmp(target->as.object.keys[i], index->as.text) == 0) {
                return nano_clone(target->as.object.values[i]);
            }
        }
        nano_fail("object key does not exist");
    }

    nano_fail("index requires List[number] or Object[text]");
    return NULL;
}

NanoValue *nano_any_field(NanoValue *target, const char *name) {
    if (!target || target->tag != NANO_OBJECT) {
        nano_fail("field access requires Object");
    }
    for (size_t i = 0; i < target->as.object.len; ++i) {
        if (strcmp(target->as.object.keys[i], name) == 0) {
            return nano_clone(target->as.object.values[i]);
        }
    }
    nano_fail("object field does not exist");
    return NULL;
}

NanoValue *nano_any_set_index(NanoValue *target, NanoValue *index, NanoValue *value) {
    NanoValue *out = nano_clone(target);
    if (!out || !index) nano_fail("set_index received invalid value");

    if (out->tag == NANO_LIST && index->tag == NANO_NUMBER) {
        double n = index->as.number;
        if (n < 0.0 || floor(n) != n || n >= (double)out->as.list.len) {
            nano_fail("assignment index out of bounds");
        }
        out->as.list.items[(size_t)n] = nano_clone(value);
        return out;
    }

    if (out->tag == NANO_OBJECT && index->tag == NANO_TEXT) {
        nano_object_put(out, index->as.text, value);
        return out;
    }

    nano_fail("indexed assignment requires List[number] or Object[text]");
    return NULL;
}

NanoValue *nano_any_set_field(NanoValue *target, const char *name, NanoValue *value) {
    NanoValue *out = nano_clone(target);
    if (!out || out->tag != NANO_OBJECT) {
        nano_fail("field assignment requires Object");
    }
    nano_object_put(out, name, value);
    return out;
}

static char *nano_to_string(const NanoValue *value);

static int nano_compare_keys(const void *lhs, const void *rhs) {
    const char *a = *(const char * const *)lhs;
    const char *b = *(const char * const *)rhs;
    return strcmp(a, b);
}

static char *nano_number_text(double value) {
    char buffer[64];
    if (isfinite(value) && floor(value) == value && fabs(value) < 9223372036854775807.0) {
        snprintf(buffer, sizeof(buffer), "%.0f", value);
    } else {
        snprintf(buffer, sizeof(buffer), "%g", value);
    }
    return nano_strdup_value(buffer);
}

static char *nano_to_string(const NanoValue *value) {
    if (!value) return nano_strdup_value("null");

    switch (value->tag) {
        case NANO_NULL:
            return nano_strdup_value("null");
        case NANO_NUMBER:
            return nano_number_text(value->as.number);
        case NANO_BOOL:
            return nano_strdup_value(value->as.boolean ? "true" : "false");
        case NANO_TEXT:
            return nano_strdup_value(value->as.text);
        case NANO_FUNCTION:
            return nano_strdup_value("<function>");
        case NANO_LIST: {
            size_t capacity = 32;
            char *out = (char *)malloc(capacity);
            if (!out) nano_fail("out of memory");
            size_t length = 0;
            out[length++] = '[';
            for (size_t i = 0; i < value->as.list.len; ++i) {
                char *part = nano_to_string(value->as.list.items[i]);
                size_t part_len = strlen(part);
                size_t need = length + part_len + 3;
                while (need >= capacity) {
                    capacity *= 2;
                    out = (char *)realloc(out, capacity);
                    if (!out) nano_fail("out of memory");
                }
                if (i > 0) out[length++] = ',';
                if (i > 0) out[length++] = ' ';
                memcpy(out + length, part, part_len);
                length += part_len;
            }
            if (length + 2 >= capacity) {
                capacity *= 2;
                out = (char *)realloc(out, capacity);
                if (!out) nano_fail("out of memory");
            }
            out[length++] = ']';
            out[length] = '\0';
            return out;
        }
        case NANO_OBJECT: {
            size_t count = value->as.object.len;
            char **sorted = (char **)malloc(count * sizeof(char *));
            if (count && !sorted) nano_fail("out of memory");
            for (size_t i = 0; i < count; ++i) sorted[i] = value->as.object.keys[i];
            qsort(sorted, count, sizeof(char *), nano_compare_keys);

            size_t capacity = 32;
            char *out = (char *)malloc(capacity);
            if (!out) nano_fail("out of memory");
            size_t length = 0;
            out[length++] = '{';

            for (size_t s = 0; s < count; ++s) {
                const char *key = sorted[s];
                size_t index = 0;
                while (index < count && strcmp(value->as.object.keys[index], key) != 0) index++;
                char *part = nano_to_string(value->as.object.values[index]);
                size_t need = length + strlen(key) + strlen(part) + 5;
                while (need >= capacity) {
                    capacity *= 2;
                    out = (char *)realloc(out, capacity);
                    if (!out) nano_fail("out of memory");
                }
                if (s > 0) out[length++] = ',';
                if (s > 0) out[length++] = ' ';
                memcpy(out + length, key, strlen(key));
                length += strlen(key);
                out[length++] = ':';
                out[length++] = ' ';
                memcpy(out + length, part, strlen(part));
                length += strlen(part);
            }

            if (length + 2 >= capacity) {
                capacity *= 2;
                out = (char *)realloc(out, capacity);
                if (!out) nano_fail("out of memory");
            }
            out[length++] = '}';
            out[length] = '\0';
            free(sorted);
            return out;
        }
    }

    return nano_strdup_value("null");
}

void nano_any_print(NanoValue *value) {
    char *text = nano_to_string(value);
    puts(text);
    free(text);
}

static NanoValue *nano_concat(const NanoValue *left, const NanoValue *right) {
    char *a = nano_to_string(left);
    char *b = nano_to_string(right);
    size_t length = strlen(a) + strlen(b);
    char *text = (char *)malloc(length + 1);
    if (!text) nano_fail("out of memory");
    memcpy(text, a, strlen(a));
    memcpy(text + strlen(a), b, strlen(b));
    text[length] = '\0';
    NanoValue *out = nano_box_text(text);
    free(text);
    free(a);
    free(b);
    return out;
}

NanoValue *nano_any_binary(NanoValue *left, int op, NanoValue *right) {
    if (!left || !right) nano_fail("binary operation received null value");

    if (op == 0) {
        if (left->tag == NANO_NUMBER && right->tag == NANO_NUMBER) {
            return nano_box_number(left->as.number + right->as.number);
        }
        if (left->tag == NANO_LIST && right->tag == NANO_LIST) {
            NanoValue *out = nano_list_new();
            for (size_t i = 0; i < left->as.list.len; ++i) nano_list_push(out, left->as.list.items[i]);
            for (size_t i = 0; i < right->as.list.len; ++i) nano_list_push(out, right->as.list.items[i]);
            return out;
        }
        if (left->tag == NANO_TEXT || right->tag == NANO_TEXT) {
            return nano_concat(left, right);
        }
        nano_fail("unsupported '+' operands");
    }

    if (op >= 1 && op <= 4) {
        if (left->tag != NANO_NUMBER || right->tag != NANO_NUMBER) {
            nano_fail("arithmetic requires Number");
        }
        double a = left->as.number;
        double b = right->as.number;
        if (op == 1) return nano_box_number(a - b);
        if (op == 2) return nano_box_number(a * b);
        if (op == 3) return nano_box_number(a / b);
        return nano_box_number(fmod(a, b));
    }

    if (op == 5 || op == 6) {
        int equal = nano_value_equal(left, right);
        return nano_box_bool(op == 5 ? equal : !equal);
    }

    if (op >= 7 && op <= 10) {
        if (left->tag != NANO_NUMBER || right->tag != NANO_NUMBER) {
            nano_fail("comparison requires Number");
        }
        double a = left->as.number;
        double b = right->as.number;
        if (op == 7) return nano_box_bool(a > b);
        if (op == 8) return nano_box_bool(a >= b);
        if (op == 9) return nano_box_bool(a < b);
        return nano_box_bool(a <= b);
    }

    if (op == 11 || op == 12) {
        int a = nano_truthy(left);
        int b = nano_truthy(right);
        return nano_box_bool(op == 11 ? (a && b) : (a || b));
    }

    nano_fail("unknown binary operator");
    return NULL;
}

NanoValue *nano_any_unary(NanoValue *value, int op) {
    if (!value) nano_fail("unary operation received null value");
    if (op == 0) {
        if (value->tag != NANO_NUMBER) nano_fail("unary '-' requires Number");
        return nano_box_number(-value->as.number);
    }
    return nano_box_bool(!nano_truthy(value));
}
