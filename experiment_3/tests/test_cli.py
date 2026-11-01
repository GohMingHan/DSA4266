import sys
import pytest
from helpers import setup
from e3.cli import main


def test_cli_train_evaluate_compare(tmp_path,monkeypatch):
    import e3.train as runner
    _,data,_=setup(tmp_path)
    real_train,real_eval=runner.train,runner.evaluate
    # Test harness explicitly admits marked fixtures; production CLI has NO bypass flag.
    monkeypatch.setattr(runner,'train',lambda a:real_train(a,allow_test_only=True))
    monkeypatch.setattr(runner,'evaluate',lambda a:real_eval(a,allow_test_only=True))
    for mode in ('concat','cross'):
        monkeypatch.setattr(sys,'argv',['pipeline.py','train','--data',str(data),'--out',str(tmp_path/mode),
            '--mode',mode,'--device','cpu','--epochs','1','--batch-size','8','--width','16','--heads','2','--head-width','12','--threads','2'])
        main()
        monkeypatch.setattr(sys,'argv',['pipeline.py','evaluate','--data',str(data),'--checkpoint',str(tmp_path/mode/'best.pt'),
            '--out',str(tmp_path/(mode+'_eval')),'--device','cpu','--threads','2'])
        main()
    monkeypatch.setattr(sys,'argv',['pipeline.py','compare','--runs',str(tmp_path/'concat_eval'),str(tmp_path/'cross_eval'),
        '--out',str(tmp_path/'comparison'),'--bootstrap','10'])
    main();assert (tmp_path/'comparison'/'comparison.csv').exists()


def test_cli_no_silent_fp16(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['pipeline.py','train','--data','unused','--out','unused','--amp','fp16'])
    with pytest.raises(SystemExit) as e:main()
    assert e.value.code==2


def test_cli_rejects_fixture_as_research(tmp_path,monkeypatch):
    _,data,_=setup(tmp_path)
    monkeypatch.setattr(sys,'argv',['pipeline.py','train','--data',str(data),'--out',str(tmp_path/'run'),'--device','cpu'])
    with pytest.raises(ValueError,match='Synthetic'):main()
    assert not (tmp_path/'run').exists()
