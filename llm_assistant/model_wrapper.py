import pickle

import stat_lm

class ModelWrapper:
    """
    Класс, который инкапсулирует всю логику генерации текста по загруженной модели и тексту.
    Тут обрабаываем подгрузку всех существующих моделей и параметров генерации под них

    load - подгрузка модели по нажатии кнопки выбора модели
    generate - генерация заданного текста текущей подгруженной моделью после команды /generate
    """
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.generate_kwargs = None

    def load(self, model_name: str, test_inference: bool = True) -> (bool, str):
        """ Load model by model_name. Return load status and error message. True if success """
        try:
            # ['StatLM', 'GPT', 'Llama']
            if model_name == 'StatLM':
                self.model, self.generate_kwargs = stat_lm.construct_model()
            else:
                return False, f"Модель {model_name} еще не поддерживается"
        except Exception as e:
            return False, f"Error while loading model {model_name}: {e}"

        if test_inference:
            try:
                result = self.model.generate("test", **self.generate_kwargs)
            except Exception as e:
                return False, f"Error while test inference model: {e}"

            if not isinstance(result, str):
                return False, f"Test inference result is not string: {type(result)}"

        self.current_model_name = model_name
        return True, ""


    def generate(self, input_text: str) -> (bool, str):
        """ generate text by context 'input_text'. Return status and message. True if success """
        if self.model is None or self.current_model_name is None:
            return False, "Need to load model"

        if not isinstance(input_text, str):
            return f"Inputs is not text: {type(input_text)}"

        result = self.model.generate(input_text, **self.generate_kwargs)
        if not isinstance(result, str):
            return False, f"Inference result is not string: {type(result)}"

        return True, result

